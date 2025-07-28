import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import re
import torch
from torchvision import transforms
from dataloader import EUVPDataSet, UIEBDataSet
from ori_model import UWnet
from fuse_model import UWnet as FuseUWnet
from step2_model import UWnet as Step2UWnet
from SSM_model import SSUWnet
from metrics_calculation import calculate_metrics_ssim_psnr, calculate_UIQM
import torchvision
from tqdm import tqdm


def get_model(model_id, num_layers, device):
    if model_id == 0:
        model = UWnet(num_layers=num_layers)
    elif model_id == 1:
        model = FuseUWnet(num_layers=num_layers)
    elif model_id == 2:
        model = Step2UWnet(num_layers=num_layers)
    elif model_id == 3:
        model = SSUWnet(num_layers=num_layers)
    else:
        raise ValueError(f"Invalid model id: {model_id}")
    return model.to(device)


@torch.no_grad()
def run_eval(model, test_loader, device, output_path, gt_path):
    model.eval()
    os.makedirs(output_path, exist_ok=True)

    for i, (img, _, name) in enumerate(tqdm(test_loader, desc="[Testing]")):
        img = img.to(device)
        output = model(img)
        torchvision.utils.save_image(output, os.path.join(output_path, name[0]))

    SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(output_path, gt_path)
    UIQM_measures = calculate_UIQM(output_path)
    print(f"[Eval] UIQM: {sum(UIQM_measures)/len(UIQM_measures):.4f}, SSIM: {sum(SSIM_measures)/len(SSIM_measures):.4f}, PSNR: {sum(PSNR_measures)/len(PSNR_measures):.4f}")


def evaluate_all_experiments(root_dir="experiment", num_layers=3, device="cuda"):
    folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    pattern = re.compile(r"(euvp|uieb)_step(\d+)")

    for folder in folders:
        match = pattern.match(folder.lower())
        if not match:
            continue

        dataset_name = match.group(1)
        model_id = int(match.group(2))
        folder_path = os.path.join(root_dir, folder)

        print(f"\n>>> Evaluating {folder} | Dataset: {dataset_name.upper()}, Model ID: {model_id}")

        # 准备模型
        model = get_model(model_id, num_layers=num_layers, device=device)
        if model_id >0:
            model = torch.nn.DataParallel(model)

        # 加载权重（默认权重路径为 snapshots/model_epoch_*.ckpt）
        snapshots_dir = os.path.join(folder_path)
        weights = sorted([f for f in os.listdir(snapshots_dir) if f.endswith(".ckpt")])
        if not weights:
            print(f"❌ No weights found in {snapshots_dir}")
            continue
        weight_path = os.path.join(snapshots_dir, weights[-1])
        model.load_state_dict(torch.load(weight_path))
        print(f"✅ Loaded weights: {weight_path}")

        # 准备数据集
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        if dataset_name == "euvp":
            test_dataset = EUVPDataSet("./data/EUVP/", transform, is_train=False, prop=0.8)
            gt_path = "./data/EUVP/GTr/"
        elif dataset_name == "uieb":
            test_dataset = UIEBDataSet("./data/UIEB/", transform, is_train=False)
            gt_path = "./data/UIEB/reference_test/"
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            continue

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=torch.cuda.device_count(), shuffle=False)

        # 评估并保存图像
        output_path = os.path.join(folder_path, "output")
        run_eval(model, test_loader, device, output_path, gt_path)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_all_experiments(device=device)
