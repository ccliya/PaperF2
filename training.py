import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置可见的GPU设备
from torch.nn import Module
import torchvision
from torchvision import transforms
import torch
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import EUVPDataSet, UIEBDataSet

from metrics_calculation import *
from ori_model import UWnet
from fuse_model import UWnet as FuseUWnet
from step2_model import UWnet as Step2UWnet
from SSM_model import SSUWnet
from combined_loss import *
import numpy as np

__all__ = [
    "Trainer",
    "setup",
    "training",
]

def create_log_folder(base_folder="log"):
    os.makedirs(base_folder, exist_ok=True)
    existing = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("exp_")]
    existing_ids = sorted([int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()])
    next_id = existing_ids[-1] + 1 if existing_ids else 1
    exp_name = f"exp_{next_id:03d}"
    exp_path = os.path.join(base_folder, exp_name)
    os.makedirs(exp_path)
    return exp_path

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module
    log_dir: str

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader=None):
        device = config.device
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []

        # 先评估初始模型
        UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        print(f"[Test][Epoch 0] UIQM: {np.mean(UIQM):.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")
        self._write_log('test_log.txt', f"Epoch 0: UIQM={np.mean(UIQM):.4f}, SSIM={np.mean(SSIM):.4f}, PSNR={np.mean(PSNR):.4f}\n")

        for epoch in trange(0, config.num_epochs, desc="[Full Loop]", leave=False):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0
            total_loss_tmp = 0

            if epoch > 1 and epoch % config.step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.7
                print(f"Learning rate decayed at epoch {epoch}")

            for inp, label, _ in tqdm(train_dataloader, desc="[Train]", leave=False):
                inp = inp.to(device)
                label = label.to(device)

                self.model.train()

                self.opt.zero_grad()
                out = self.model(inp)
                loss, mse_loss, vgg_loss = self.loss(out, label)

                loss.backward()
                self.opt.step()

                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                total_loss_tmp += loss.item()

            avg_total_loss = total_loss_tmp / len(train_dataloader)
            avg_vgg_loss = vgg_loss_tmp / len(train_dataloader)
            avg_primary_loss = primary_loss_tmp / len(train_dataloader)

            total_loss_lst.append(avg_total_loss)
            vgg_loss_lst.append(avg_vgg_loss)
            primary_loss_lst.append(avg_primary_loss)

            print(f"[Train][Epoch {epoch}] Total Loss: {avg_total_loss:.6f}, Primary Loss: {avg_primary_loss:.6f}, VGG Loss: {avg_vgg_loss:.6f}")
            self._write_log('train_log.txt', f"Epoch {epoch}: Total Loss={avg_total_loss:.6f}, Primary Loss={avg_primary_loss:.6f}, VGG Loss={avg_vgg_loss:.6f}\n")

            if (config.test == True) and (epoch % config.eval_steps == 0):
                UIQM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                print(f"[Test][Epoch {epoch}] UIQM: {np.mean(UIQM):.4f}, SSIM: {np.mean(SSIM):.4f}, PSNR: {np.mean(PSNR):.4f}")
                self._write_log('test_log.txt', f"Epoch {epoch}: UIQM={np.mean(UIQM):.4f}, SSIM={np.mean(SSIM):.4f}, PSNR={np.mean(PSNR):.4f}\n")

            if epoch % config.print_freq == 0:
                print('Epoch [{}/{}], Total Loss: {:.6f}, Primary Loss: {:.6f}, VGG Loss: {:.6f}'.format(
                    epoch, config.num_epochs, avg_total_loss, avg_primary_loss, avg_vgg_loss))

            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if epoch % config.snapshot_freq == 0:
                save_path = os.path.join(config.snapshots_folder, f'model_epoch_{epoch}.ckpt')
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for i, (img, _, name) in enumerate(test_dataloader):
            img = img.to(config.device)
            generate_img = test_model(img)
            torchvision.utils.save_image(generate_img, os.path.join(config.output_images_path, name[0]))
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path, config.test_path)
        UIQM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, SSIM_measures, PSNR_measures

    def _write_log(self, filename, content):
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, 'a') as f:
            f.write(content)


def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    if config.model == 0:
        model = UWnet(num_layers=config.num_layers).to(config.device)
    elif config.model == 1:
        model = FuseUWnet(num_layers=config.num_layers).to(config.device)
    elif config.model == 2:
        model = Step2UWnet(num_layers=config.num_layers).to(config.device)
    elif config.model == 3:
        model = SSUWnet(num_layers=config.num_layers).to(config.device)
    model = torch.nn.DataParallel(model)  # 使用DataParallel支持多GPU训练
    transform = transforms.Compose([transforms.Resize((config.resize, config.resize)), transforms.ToTensor()])

    if config.data == "euvp":
        train_dataset = EUVPDataSet(config.euvp_data, transform, is_train=True, prop=0.8)
        test_dataset = EUVPDataSet(config.euvp_data, transform, is_train=False, prop=0.8)
        test_path = config.euvp_data + 'GTr/'
    elif config.data == "ueib":
        train_dataset = UIEBDataSet(config.uieb_data, transform, is_train=True)
        test_dataset = UIEBDataSet(config.uieb_data, transform, is_train=False)
        test_path = config.uieb_data + 'reference_test/'
    else:
        raise ValueError("Unknown dataset name in config.data")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    print("Train Dataset Reading Completed.")

    loss = combinedloss(config)
    if config.model == 3:
        loss = CombinedLoss2(config)
    
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(model, opt, loss, config.log_dir)

    # 方便eval时读取
    config.test_path = test_path

    return train_dataloader, test_dataloader, model, trainer


def training(config):
    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, config, ds_test)
    print("==================")
    print("Training complete!")
    print("==================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--data', type=str, default="ueib", help='path of input images default:./data/')
    parser.add_argument('--model', type=int, default=0, help='model step')
    parser.add_argument('--euvp_data', type=str, default="./data/EUVP/", help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--uieb_data', type=str, default="./data/UIEB/", help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size', type=int, default=400, help="Period of learning rate decay")  # 50
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--test_batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--snapshot_freq', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)

    # 下面不再用单独路径，而由log_dir自动管理
    config = parser.parse_args()
    device_num = torch.cuda.device_count()
    config.train_batch_size = config.train_batch_size * device_num
    config.test_batch_size = config.test_batch_size * device_num
    # 创建log/exp_XXX文件夹
    log_dir = create_log_folder()
    config.log_dir = log_dir

    # 设置快照和输出图像文件夹
    config.snapshots_folder = os.path.join(log_dir, "snapshots")
    config.output_images_path = os.path.join(log_dir, "output_images")
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.output_images_path, exist_ok=True)

    # 保存配置到 config.txt
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        for k, v in vars(config).items():
            f.write(f"{k}: {v}\n")

    training(config)