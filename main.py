"""
Pytorch implementation of Proto-Caps in paper
    Interpretable Medical Image Classification using Prototype Learning and Privileged Information .

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""

import sys
import torch
import datetime
import numpy as np
from models import ProtoCapsNet
from data_loader import load_lidc
from lidc_semantics_cn import explain_attribute, explain_malignancy, get_attribute_cn  #ou
from push import pushprotos
from train import train_model
from test import test, test_indepth, test_indepth_attripredCorr, test_show
import nibabel as nib
import torch.nn.functional as F


def center_crop_or_pad(vol, target_shape):
    # vol: ndarray (D,H,W) or (H,W) ; target_shape: list/tuple
    vol = np.asarray(vol)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]  # make (1,H,W)
    d, h, w = vol.shape
    td, th, tw = (target_shape if len(target_shape) == 3 else (1, target_shape[0], target_shape[1]))
    out = np.zeros((td, th, tw), dtype=vol.dtype)
    sd = max(0, (d - td)//2); sh = max(0, (h - th)//2); sw = max(0, (w - tw)//2)
    ed = sd + min(td, d); eh = sh + min(th, h); ew = sw + min(tw, w)
    od = max(0, (td - d)//2); oh = max(0, (th - h)//2); ow = max(0, (tw - w)//2)
    out[od:od+(ed-sd), oh:oh+(eh-sh), ow:ow+(ew-sw)] = vol[sd:ed, sh:eh, sw:ew]
    return out

def load_nii_as_input(patch_path, resize_shape, threeD):
    # load nii and prepare tensor for model
    nii = nib.load(patch_path)
    vol = nii.get_fdata()  # (Z,Y,X) or (Y,X) depending file
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]  # (1,H,W)
    if not threeD:
        zc = vol.shape[0] // 2
        slice2d = vol[zc]  # (H,W)
        t = torch.tensor(slice2d).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=tuple(resize_shape), mode='bilinear', align_corners=False)
        return t  # (1,1,H,W)
    else:
        # expect resize_shape length 3, else default to 48 cube
        target = resize_shape if len(resize_shape) == 3 else [48,48,48]
        cropped = center_crop_or_pad(vol, target)  # (D,H,W)
        t = torch.tensor(cropped).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        # convert to channels-first 3D conv shape if model expects (B,C,D,H,W)
        return t

def prototypeLearningStatus(model, unfreeze):
    for protoi in range(len(model.protodigis_list)):
        model.protodigis_list[protoi].requires_grad = unfreeze

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ProtoCaps")
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.02, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lam_recon', default=0.512, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--warmup', default=100, type=int,
                        help="Number of epochs before prototypes are fitted.")
    parser.add_argument('--push_step', default=10, type=int,
                        help="Prototypes are pushed every [push_step] epoch.")
    parser.add_argument('--split_number', default=0, type=int)
    parser.add_argument('--shareAttrLabels', default=1.0, type=float)
    parser.add_argument('--threeD', default=False, type=bool)
    parser.add_argument('--resize_shape', nargs='+', type=int, default=[32,32],  # for 3D experiments:[48, 48, 48]
                        help="Size of boxes cropped out of CT volumes as model input")
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--epoch', type=int, help="Set the epoch of chosen model")
    parser.add_argument('--infer_patch', type=str, default=None, help="Path to single patch .nii or .nii.gz for inference")
    # small out_dim_caps leads to different prototypes per attribute class
    parser.add_argument('--out_dim_caps', type=int, default=16, help="Set dimension of output capsule vectors.")
    parser.add_argument('--num_protos', type=int, default=2, help="Set number of prototypes per attribute class")
    parser.add_argument('--numcaps', type=int, default=8, help="Number of attribute capsules (use 8 for this repo)")
    args = parser.parse_args()
    print(args)

    # single patch inference mode
    if args.infer_patch is not None:
        if args.model_path is None:
            raise TypeError("Please specify --model_path for inference")
        map_loc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded = torch.load(args.model_path, map_location=map_loc)

        # if loaded is a state_dict, rebuild model with provided numcaps and load weights
        if isinstance(loaded, dict) and not hasattr(loaded, "forward"):
            print("Loaded object looks like state_dict. Rebuilding ProtoCapsNet with numcaps =", args.numcaps)
            model = ProtoCapsNet(input_size=[1, *args.resize_shape],
                                 numcaps=args.numcaps,
                                 routings=args.routings,
                                 out_dim_caps=args.out_dim_caps,
                                 activation_fn="sigmoid",
                                 threeD=args.threeD,
                                 numProtos=args.num_protos)
            model.load_state_dict(loaded)
        else:
            model = loaded

        model.to(map_loc)
        model.eval()

        inp = load_nii_as_input(args.infer_patch, args.resize_shape, args.threeD)
        inp = inp.to(map_loc)

        with torch.no_grad():
            out = model(inp)

        print("Model Outputs:")
        if isinstance(out, dict):
            for key, value in out.items():
                print(f"{key}: {value.cpu().detach().numpy()}")

        def _dbg_out(o):
            print("Model output type:", type(o))
            if torch.is_tensor(o):
                print("  tensor shape:", tuple(o.shape))
            elif isinstance(o, (list, tuple)):
                print("  list/tuple len:", len(o))
                for i, e in enumerate(o):
                    if torch.is_tensor(e):
                        print(f"   [{i}] tensor shape={tuple(e.shape)}")
                    else:
                        print(f"   [{i}] type={type(e)}")
            elif isinstance(o, dict):
                print("  dict keys:", list(o.keys()))
                for k, v in o.items():
                    if torch.is_tensor(v):
                        print(f"   key='{k}' tensor shape={tuple(v.shape)}")
            else:
                print("  repr:", repr(o)[:200])

        # parse common output formats
        attr_logits = None
        mal_logits = None

        if isinstance(out, dict):
            attr_logits = out.get('attr', None)
            mal_logits = out.get('mal', None)
        elif isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, dict):
                    attr_logits = item.get('attr', attr_logits)
                    mal_logits = item.get('mal', mal_logits)
                elif torch.is_tensor(item) and item.shape[-1] == 8:
                    attr_logits = item
                elif torch.is_tensor(item) and item.shape[-1] == 5:
                    mal_logits = item
        elif torch.is_tensor(out):
            attr_logits = out

        # 属性处理（使用 lidc_semantics_cn 模块）
        if attr_logits is not None:
            attr_probs = torch.sigmoid(attr_logits.squeeze()).cpu().numpy()
            attr_names = [
                "Subtlety", "Internal Structure", "Calcification", "Sphericity",
                "Margin", "Lobulation", "Spiculation", "Texture"
            ]

            # print("🔍 视觉属性预测:")
            # for i, (name, prob) in enumerate(zip(attr_names, attr_probs)):
            #     score, label = explain_attribute(name, prob)
            #     print(f"  {i+1}. {name}: 概率={prob:.4f} → 分值={score} → {label}")
            print("🔍 视觉属性预测结果：\n")
            for i, (name, prob) in enumerate(zip(attr_names, attr_probs)):
                score, label = explain_attribute(name, prob)
                name_zh = get_attribute_cn(name)
                print(f"{i+1}. {name_zh}（{name}）：{label}（分值 = {score} / 概率 = {prob:.4f}）")

        else:
            print("未能获取视觉属性预测结果。")

        # 癌样等级处理（使用 lidc_semantics_cn 模块）
        if mal_logits is not None:
            mal_probs = torch.softmax(mal_logits.squeeze(), dim=0).cpu().numpy()
            mal_score = sum((i + 1) * mal_probs[i] for i in range(5))  # 加权平均

            print(f"\n🧪 癌样等级预测分数（加权平均）: {mal_score:.4f}")
            print(f"→ 语义判断：{explain_malignancy(mal_score)}")

            print("\n癌样等级分布（LIDC 原始语义）:")
            for i, p in enumerate(mal_probs):
                label = explain_malignancy(i + 1)
                print(f"  等级 {i+1}: {p:.6f} → {label}")
        else:
            print("未能获取癌样等级预测结果。")

        sys.exit(0)


    if (not args.train and not args.test) or (args.train and args.test):
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")

    train_loader, val_loader, test_loader = load_lidc(batch_size=args.batch_size,
                                                      resize_shape=args.resize_shape,
                                                      threeD=args.threeD,
                                                      splitnumber=args.split_number)

    numattributes = next(iter(train_loader))[2].shape[1]
    print("#attributes=#caps : " + str(numattributes))


    if args.test:
        if args.model_path == None:
            raise TypeError("Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        if args.epoch == None:
            raise TypeError("Please specify the epoch of chosen model by setting the parameter --epoch=[int]")
        path = args.model_path
        model = torch.load(path)

        test_acc, test_attracc, test_dc = test_indepth(testmodel=model,
                                                       data_loader=test_loader,
                                                       epoch=args.epoch,
                                                       prototypefoldername=path.split("_")[0])
        print("PE_test_acc (Testing accuracy with use of prototypes): " + str(test_acc))
        print("PE_test_attr_acc: " + str(test_attracc))
        print("dc without exchange:" + str(test_dc))
        _, test_acc, te_attr_acc = test(testmodel=model, data_loader=test_loader, arguments=args)
        print('test acc = %.4f (within 1 score)' % test_acc)
        print("attr test acc = " + str(te_attr_acc))

        test_indepth_attripredCorr(testmodel=model,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   epoch=args.epoch,
                                   prototypefoldername=path.split("_")[0])

        # test_show(testmodel=model, data_loader=test_loader,
        #                            epoch=args.epoch,
        #                            prototypefoldername=path.split("_")[0])

    if args.train:
        model = ProtoCapsNet(input_size=[1, *args.resize_shape], numcaps=numattributes, routings=3,
                             out_dim_caps=args.out_dim_caps, activation_fn="sigmoid", threeD=args.threeD, numProtos=args.num_protos)
        model.cuda()
        print(model)

        opt_specs = [{'params': model.conv1.parameters(), 'lr': args.lr},
                     {'params': model.primarycaps.parameters(), 'lr': args.lr},
                     {'params': model.digitcaps.parameters(), 'lr': args.lr},
                     {'params': model.decoder.parameters(), 'lr': args.lr},
                     {'params': model.predOutLayers.parameters(), 'lr': args.lr},
                     {'params': model.relu.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer0.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer1.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer2.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer3.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer4.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer5.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer6.parameters(), 'lr': args.lr},
                     {'params': model.attrOutLayer7.parameters(), 'lr': args.lr},
                     {'params': model.protodigis0, 'lr': 0.5},
                     {'params': model.protodigis1, 'lr': 0.5},
                     {'params': model.protodigis2, 'lr': 0.5},
                     {'params': model.protodigis3, 'lr': 0.5},
                     {'params': model.protodigis4, 'lr': 0.5},
                     {'params': model.protodigis5, 'lr': 0.5},
                     {'params': model.protodigis6, 'lr': 0.5},
                     {'params': model.protodigis7, 'lr': 0.5}]
        optimizer = torch.optim.Adam(opt_specs)

        print("training samples: " + str(len(train_loader.dataset)))
        print("val samples: " + str(len(val_loader.dataset)))
        print("test samples: " + str(len(test_loader.dataset)))
        train_samples_with_attrLabels_Loss = torch.randperm(len(train_loader.dataset))[
                                             :int(args.shareAttrLabels * len(train_loader.dataset))]
        print(str(len(train_samples_with_attrLabels_Loss)) + " samples are being considered of having attribute labels")

        best_val_acc = 0.0
        #datestr = str(datetime.datetime.now())  #ou ori
        datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  #ou
        print("this run has datestr " + datestr)
        protosavedir = "./prototypes/" + str(datestr)
        #os.mkdir(protosavedir)  #ou ori
        os.makedirs(protosavedir, exist_ok=True)  #ou

        earlyStopping_counter = 1
        earlyStopping_max = 10  # push iterations
        for ep in range(args.epochs):
            if ep % args.push_step == 0:
                if ep >= args.warmup:
                    print("Pushing")
                    model, mindists_X, mindists_attr_sc = pushprotos(model_push=model, data_loader=train_loader,
                                                                     idx_with_attri=train_samples_with_attrLabels_Loss)
                    protosavedir = "./prototypes/" + str(datestr) + "/" + str(ep)
                    os.mkdir(protosavedir)
                    for cpsi in range(len(mindists_X)):
                        for proto_idx in range(mindists_X[cpsi].shape[0]):
                            for proto_idx2 in range(mindists_X[cpsi].shape[1]):
                                np.save(os.path.join(protosavedir + "/",
                                                     "cpslnr" + str(cpsi) + "_protonr" + str(
                                                         proto_idx) + "-" + str(proto_idx2) + "_gtattrcs" + str(
                                                         mindists_attr_sc[cpsi][proto_idx, proto_idx2])),
                                        mindists_X[cpsi][proto_idx, proto_idx2, 0])

                    valwProtoE_acc, valwProtoE_attracc, _ = test_indepth(testmodel=model,
                                                                         data_loader=val_loader,
                                                                         epoch=ep,
                                                                         prototypefoldername=datestr)
                    print("PE_val_acc: " + str(valwProtoE_acc))
                    print("PE_val_attracc: " + str(valwProtoE_attracc))

                    if valwProtoE_acc > best_val_acc:
                        torch.save(model, str(datestr) + "_" + str(valwProtoE_acc) + "_" + str(ep) + '.pth')
                        print("Save new best model with path: " + str(datestr) + "_" + str(valwProtoE_acc) + "_" + str(ep) + '.pth')
                        best_val_acc = valwProtoE_acc
                        earlyStopping_counter = 1
                    else:
                        earlyStopping_counter += 1
                        if earlyStopping_counter > earlyStopping_max:
                            sys.exit()

            if ep < args.warmup:
                prototypeLearningStatus(model, False)
            else:
                prototypeLearningStatus(model, True)
            print("Training")
            model, tr_acc, tr_attr_acc = train_model(
                model, train_loader, args, epoch=ep, optim=optimizer,
                idx_with_attri=train_samples_with_attrLabels_Loss)
            print('train acc = %.4f (within 1 score)' % tr_acc)

            print("Validation")
            val_loss, val_acc, _ = test(testmodel=model, data_loader=val_loader, arguments=args)
            print('val acc = %.4f (within 1 score), val loss = %.5f' % (
                val_acc, val_loss))

            print("Epoch " + str(ep) + ' ' + '-' * 70)

