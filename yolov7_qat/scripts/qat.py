################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import sys
import os

# Add the current directory to PYTHONPATH for YoloV7
sys.path.insert(0, os.path.abspath("."))
pydir = os.path.dirname(__file__)

import yaml
import collections
import warnings
import argparse
import json
from pathlib import Path
from datetime import datetime

# ONNX
import onnx

# PyTorch
import torch
import torch.nn as nn

# YoloV7
import test
from models.yolo import Model
from models.common import Conv
from utils.datasets import create_dataloader
from utils.google_utils import attempt_download
from utils.general import init_seeds
from utils.general import check_img_size
## Yolov7 End2End
from models.experimental import End2End

import quantization.quantize as quantize

from pycocotools_anno_json import build_pycocotools_anno_json

# Disable all warning
warnings.filterwarnings("ignore")


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def print_map_scores(data):
   
    # Extract mAP scores
    origin_map = data[0][1]
    ptq_map = data[1][1]
    qat_maps = [(entry[0], entry[1]) for entry in data[2:]]
    
    # Find the best QAT score
    best_qat = max(qat_maps, key=lambda x: x[1])
    
    # Print the results
    print("Origin mAP@.5:.95:", origin_map)
    print("PTQ mAP@.5:.95:", ptq_map)
    print("Best QAT mAP@.5:.95:", best_qat[0], best_qat[1])


# Load YoloV7 Model
def load_yolov7_model(weight, device) -> Model:

    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model

def generate_custom_annotation_json(data, use_pycocotools):
    # Load YAML file
    with open(data, 'r') as f:
        data_yaml = yaml.load(f, Loader=yaml.SafeLoader) 
    val_root_dir = os.path.join(os.path.dirname(data_yaml['val']), 'annotations_coco')

    os.makedirs(val_root_dir, exist_ok=True)

    # Create the JSON file
    annotations_json=os.path.join(val_root_dir,'custom_dataset_annotation.json')
    if use_pycocotools:
        # Call the function to create instances JSON
        if os.path.exists(annotations_json):
            print(f"Skipping generation of COCO format annotation file. The file {annotations_json} already exists.")
            return True
        else:
            print(f"Generating COCO format annotation file: {annotations_json}")
            
            if not (build_pycocotools_anno_json(data, annotations_json)):
                return False
        return True
    else:
        if os.path.exists(annotations_json):
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            annotations_json_old = os.path.join(val_root_dir, f'custom_dataset_annotation_{current_time}.json')
            os.rename(annotations_json, annotations_json_old)
        return False



def create_train_dataloader(train_path, img_size, batch_size,hyp_path):

    with open(hyp_path) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    loader = create_dataloader(
        train_path, 
        imgsz=img_size, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=False, cache=False, stride=32,pad=0, image_weights=False)[0]
    return loader



def create_val_dataloader(test_path, img_size, batch_size, keep_images=None):

    loader = create_dataloader(
        test_path, 
        imgsz=img_size, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False,stride=32,pad=0.5, image_weights=False)[0]

    def subclass_len(self):
        if keep_images is not None:
            return keep_images
        return len(self.img_files)

    loader.dataset.__len__ = subclass_len
    return loader




def evaluate_dataset(model, dataloader, data, using_cocotools = False, is_coco=False, save_dir=".", conf_thres=0.001, iou_thres=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        data, 
        save_dir=Path(save_dir),
        dataloader=dataloader, conf_thres=conf_thres,iou_thres=iou_thres,model=model,is_coco=is_coco,
        plots=False,half_precision=True,save_json=using_cocotools)[0][3]


def export_onnx(model : Model, file, img_size=640, dynamic_batch=False, end2end=False, topk_all=100, simplify=False, iou_thres=0.65, conf_thres=0.45 ):
    
    device = next(model.parameters()).device
    model.float()

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    image_check=[img_size,img_size]
    img_size, _ = [check_img_size(x, gs) for x in image_check]  # verify img_size are gs-multiples

    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    
    
    if not end2end:
        model.model[-1].concat = True
        grid_old_func = model.model[-1]._make_grid
        model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())
        quantize.export_onnx(model, dummy, file, opset_version=13, 
            input_names=["images"], output_names=["outputs"], 
            dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
        )
        model.model[-1].concat = False
        model.model[-1]._make_grid = grid_old_func
    
    if end2end and dynamic_batch:
        print(model.model[-1].concat)
        model.model[-1].export = False  # set Detect() layer grid export
        
        grid_old_func = model.model[-1]._make_grid
        model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())

        labels = model.names
        batch_size = 'batch'

        dynamic_axes = {
                'images': {
                    0: 'batch',
                }, }

        output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
        dynamic_axes.update(output_axes)

        print('\nStarting export end2end onnx model for TensorRT...')
        model = End2End(model,topk_all,iou_thres,conf_thres,None,device,len(labels))
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        shapes = [batch_size, 1, batch_size, topk_all, 4,
                    batch_size, topk_all, batch_size, topk_all]
                
        quantize.export_onnx(model, dummy, file, opset_version=13, 
            input_names=["images"], output_names=output_names, 
            dynamic_axes=dynamic_axes
        )
        onnx_model = onnx.load(file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
        
        if simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(onnx_model,file)
        print('ONNX export success, saved as %s' % file)





def cmd_quantize(weight, data, img_size, batch_size, hyp, device, ignore_policy, experiment, project_name, save_ptq, save_qat, supervision_stride, iters, eval_origin, eval_ptq, use_pycocotools):
    
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = data.endswith('coco.yaml')

    ## build coco annotation 
    if not is_coco and use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, True)
    if not use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, False)
    
    using_cocotools=False
    if is_coco or use_pycocotools:
        using_cocotools=True

    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check

    train_path = data_dict['train']
    test_path = data_dict['val']

    save_dir=os.path.join(experiment, project_name)

    os.makedirs(save_dir, exist_ok=True)

    save_ptq = os.path.join(save_dir,save_ptq)
    save_qat = os.path.join(save_dir,save_qat)
    
    quantize.initialize()

    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    img_check=[img_size,img_size]
    img_size, _ = [check_img_size(x, gs) for x in img_check]  # verify img_size are gs-multiples
    
    train_dataloader = create_train_dataloader(train_path,img_size,batch_size, hyp)
    val_dataloader   = create_val_dataloader(test_path,img_size, batch_size)
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    quantize.calibrate_model(model, train_dataloader, device)

    summary_file = os.path.join(save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_dataset(model, val_dataloader, data, using_cocotools = using_cocotools, is_coco=is_coco, save_dir=save_dir )
            summary.append(["Origin", ap])

    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_dataset(model, val_dataloader, data, using_cocotools = using_cocotools, is_coco=is_coco, save_dir=save_dir )
        summary.append(["PTQ", ap])

    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    best_ap = 0
    def per_epoch(model, epoch, lr):

        nonlocal best_ap
        ap = evaluate_dataset(model, val_dataloader, data, using_cocotools = using_cocotools, is_coco=is_coco, save_dir=save_dir )
        summary.append([f"QAT{epoch}", ap])
        print_map_scores(summary)
        if ap > best_ap:
            print(f"Save qat model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)

    def preprocess(datas):
        return datas[0].to(device).float() / 255.0

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    quantize.finetune(
        model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, supervision_policy=supervision_policy())


def cmd_export(weight, experiment, project_name, save, img_size, dynamic, end2end, topk_all, simplify, iou_thres, conf_thres):
    quantize.initialize()
    
    save_dir = os.path.join(experiment, project_name)
    os.makedirs(save_dir, exist_ok=True)

    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(save_dir, name + ".onnx")
    else:
        save = os.path.join(save_dir,save)

    export_onnx(torch.load(weight, map_location="cpu")["model"],  save, img_size, dynamic_batch=dynamic, end2end=end2end, topk_all=topk_all, simplify=simplify, iou_thres=iou_thres, conf_thres=conf_thres)
    print(f"Save onnx to {save}")

def cmd_sensitive_analysis(weight, device, data, img_size, batch_size, hyp, experiment, project_name, summary_save, num_image, use_pycocotools):
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = data.endswith('coco.yaml')


    save_dir = os.path.join(experiment, project_name)
    os.makedirs(save_dir, exist_ok=True)

    summary_save=os.path.join(save_dir, summary_save)
    ## build coco annotation 
    if not is_coco and use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, True)
    if not use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, False)
    
    using_cocotools=False
    if is_coco or use_pycocotools:
        using_cocotools=True
        
    
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check

    train_path = data_dict['train']
    test_path = data_dict['val']

    quantize.initialize()
    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    img_check=[img_size,img_size]
    img_size, _ = [check_img_size(x, gs) for x in img_check]  # verify img_size are gs-multiples
    
    train_dataloader = create_train_dataloader(train_path,img_size,batch_size, hyp)
    val_dataloader   = create_val_dataloader(test_path,img_size, batch_size, keep_images=None if num_image is None or num_image < 1 else num_image )

    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_dataset(model, val_dataloader, data, using_cocotools=using_cocotools, is_coco=is_coco)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_dataset(model, val_dataloader, data, using_cocotools=using_cocotools, is_coco=is_coco)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


def cmd_test(weight, device, data, img_size, batch_size, confidence, nmsthres, use_pycocotools):
    img_size.extend([img_size[-1]] * (2 - len(img_size))) # extend to 2 sizes (train, test)
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = data.endswith('coco.yaml')
    
    ## build coco annotation 
    if not is_coco and use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, True)
    if not use_pycocotools:
        use_pycocotools = generate_custom_annotation_json(data, False)
    
    using_cocotools=False
    if is_coco or use_pycocotools:
        using_cocotools=True

    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check

    train_path = data_dict['train']
    test_path = data_dict['val']

    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    img_check=[img_size,img_size]
    img_size, _ = [check_img_size(x, gs) for x in img_check]  # verify img_size are gs-multiples

    val_dataloader = create_val_dataloader(test_path,img_size, batch_size)
    evaluate_dataset(model, val_dataloader, data,  using_cocotools=using_cocotools, is_coco=is_coco, conf_thres=confidence, iou_thres=nmsthres)



if __name__ == "__main__":
    project_name = datetime.now().strftime("%Y%m%d%H%M%S")
    
    parser = argparse.ArgumentParser(prog='qat.py')
    subps  = parser.add_subparsers(dest="cmd")
    exp    = subps.add_parser("export", help="Export weight to onnx file")
    exp.add_argument("weight", type=str, default="yolov7.pt", help="export pt file")
    exp.add_argument("--save", type=lambda x: os.path.basename(x), required=False, help="export onnx file")
    exp.add_argument('--experiment', default='experiments/export/', help='save to project name')
    exp.add_argument('--project_name', default=project_name, help='save to project/name')
    exp.add_argument('--img-size', type=int, default=640, help='image sizes same for train and test')
    exp.add_argument("--dynamic", action="store_true", help="export dynamic batch")
    ### added end2end
    exp.add_argument('--end2end', action='store_true', help='export end2end onnx')
    exp.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    exp.add_argument('--simplify', action='store_true', help='simplify onnx model')
    exp.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    exp.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')

    qat = subps.add_parser("quantize", help="PTQ/QAT finetune ...")
    qat.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    qat.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    qat.add_argument('--batch-size', type=int, default=10, help='total batch size')
    qat.add_argument('--img-size', type=int, default=640, help='image sizes same for train and test')
    qat.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument("--ignore-policy", type=str, default="model\.105\.m\.(.*)", help="regx")
    qat.add_argument('--experiment', default='experiments/qat/', help='save to project name')
    qat.add_argument('--project_name', default=project_name, help='save to project/name')
    qat.add_argument("--ptq", type=lambda x: os.path.basename(x), default="ptq.pt", help="PQT Filename")
    qat.add_argument("--qat", type=lambda x: os.path.basename(x), default="qat.pt", help="PQT Filename")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--iters", type=int, default=200, help="iters per epoch")
    qat.add_argument("--eval-origin", action="store_true", help="do eval for origin model")
    qat.add_argument("--eval-ptq", action="store_true", help="do eval for ptq model")
    qat.add_argument("--use-pycocotools", action="store_true", help="Generate COCO annotation format json for the custom dataset")
    

    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    sensitive.add_argument('--batch-size', type=int, default=10, help='total batch size')
    sensitive.add_argument('--img-size', type=int, default=640, help='image sizes same for train and test')
    sensitive.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    sensitive.add_argument("--summary", type=lambda x: os.path.basename(x), default="sensitive-summary.json", help="summary save file")
    sensitive.add_argument('--experiment', default='experiments/sensitive/', help='save to project name')
    sensitive.add_argument('--project_name', default=project_name, help='save to project/name')
    sensitive.add_argument("--num-image", type=int, default=None, help="number of image to evaluate")
    sensitive.add_argument("--use-pycocotools", action="store_true", help="Generate COCO annotation json format for the custom dataset")


    testcmd = subps.add_parser("test", help="Do evaluate")
    testcmd.add_argument("weight", type=str, default="yolov7.pt", help="weight file")
    testcmd.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    testcmd.add_argument('--batch-size', type=int, default=10, help='total batch size')
    testcmd.add_argument('--img-size', type=int, default=640, help='image sizes same for train and test')
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")
    testcmd.add_argument("--use-pycocotools", action="store_true", help="Generate COCO annotation json format for the custom dataset")


    args = parser.parse_args()
    init_seeds(57)
    if args.cmd == "export":
        cmd_export(args.weight, args.experiment, args.project_name, args.save, args.img_size, args.dynamic, args.end2end, args.topk_all, args.simplify, args.iou_thres, args.conf_thres)
    elif args.cmd == "quantize":
        print(args)
        cmd_quantize(
            args.weight, args.data, args.img_size, args.batch_size, 
            args.hyp, args.device, args.ignore_policy, args.experiment, args.project_name,
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq, args.use_pycocotools
        )
    elif args.cmd == "sensitive":
        cmd_sensitive_analysis(args.weight, args.device, args.data, args.img_size, args.batch_size, args.hyp, args.experiment, args.project_name, args.summary, args.num_image, args.use_pycocotools)
    elif args.cmd == "test":
        cmd_test(args.weight, args.device, args.data, args.img_size, args.batch_size, args.confidence, args.nmsthres, args.use_pycocotools)
    else:
        parser.print_help()