import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
# >>> ADD
# from tsne import visualize_tsne_panels
from tsne import visualize_pair_tsne
from tsne import visualize_pair_umap_two_clusters
import copy

def save_model(model):
    model_path = r'../output/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path+'model.pt')
    new_model = torch.load(model_path + 'model.pt')
    return new_model

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, data_name, split, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]
        self.use_ld = config['SOLVER']["USE_LD"]

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"] + f'{data_name}/{split}/'

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if self.use_ld:
                if self.current_epoch % self.decay_interval == 0:
                    self.optim.param_groups[0]['lr'] *= self.lr_decay

            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()

        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }

        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch
    

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)

        # 原本的 df 没被真正用到可视化，这里保留不动
        df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}

        # >>> ADD: 为 t-SNE 可视化准备的收集容器
        tsne_drug_feats = []     # v_d_I
        tsne_prot_feats = []     # v_p_I
        tsne_pair_feats = []     # f
        tsne_y_true = []         # 真实标签
        tsne_y_pred_label = []   # 二分类标签（阈值后）
        

        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels) in enumerate(data_loader):
                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)

                if dataloader == "val":
                    # 验证集：照旧（不收集 t-SNE，保持轻量）
                    v_d_I, v_p_I, f, score = self.model(v_d, v_p)
                elif dataloader == "test":
                    # 测试集：使用 best_model
                    v_d_I, v_p_I, f, score = self.best_model(v_d, v_p)

                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()

                # 这里的 n 应该是概率/分数（binary_cross_entropy 通常返回 sigmoid 之后的 n）
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

                if dataloader == 'test':
                    # 你原来为了输出 CSV 保留了输入特征，这里不动
                    df['drug'] = df['drug'] + v_d.to('cpu').tolist()
                    df['protein'] = df['protein'] + v_p.to('cpu').tolist()

                    # >>> ADD: 收集可视化需要的中间表示（注意是 *融合后的* v_d_I/v_p_I 和 pair f）
                    tsne_drug_feats.append(v_d_I.detach().cpu())
                    tsne_prot_feats.append(v_p_I.detach().cpu())
                    tsne_pair_feats.append(f.detach().cpu())
                    tsne_y_true.append(labels.detach().cpu().long())

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, precision_score
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            try:
                precision = tpr / (tpr + fpr)
            except RuntimeError:
                raise ('RuntimeError: the divide==0')
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (np.array(y_pred) >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

            precision1 = precision_score(y_label, y_pred_s)

            # 输出预测 CSV（保持你原有逻辑）
            df = pd.DataFrame({
                "y_label": y_label,
                "y_pred": y_pred
            })
            df.to_csv("prediction_results.csv", index=False)

            # >>> ADD: 把收集到的 t-SNE 相关张量拼接，并画图保存
            import torch as _torch
            tsne_drug_feats = _torch.cat(tsne_drug_feats, dim=0).numpy()     # [N, D]
            # tsne_prot_feats = _torch.cat(tsne_prot_feats, dim=0).numpy()     # [N, D]
            tsne_pair_feats = _torch.cat(tsne_pair_feats, dim=0).numpy()     # [N, 2D]
            tsne_y_true = _torch.cat(tsne_y_true, dim=0).numpy()             # [N]
            tsne_y_pred_label = np.array(y_pred_s)                           # [N]

            # 输出路径：落在你的 result 输出目录里
            tsne_png = os.path.join(self.output_dir, f"tsne_panels_best_epoch_{self.best_epoch}.png")
            os.makedirs(self.output_dir, exist_ok=True)

            # 直接调用 tsne.py 里的三联图函数
            pair_feats_np = tsne_pair_feats.astype(np.float32)   # ✅
            y_true_np = np.array(y_label, dtype=int)
            y_pred_label_np = np.array(y_pred_s, dtype=int)

            tsne_png = os.path.join(self.output_dir, f"tsne_pairs_best_epoch_{self.best_epoch}.png")
            os.makedirs(self.output_dir, exist_ok=True)
            visualize_pair_tsne(
                pair_feats=pair_feats_np,
                y_true=y_true_np,
                y_pred_label=y_pred_label_np,
                save_path=tsne_png,
                max_per_class=4000,      # 数据很大可以再小点
                random_state=42,
                perplexity=None          # 让函数按规模自适应
            )
            # visualize_pair_umap_two_clusters(
            #     pair_feats=pair_feats_np,
            #     y_true=y_true_np,
            #     y_pred_label=y_pred_label_np,
            #     save_path=os.path.join(self.output_dir, f"pairs_umap_two_clusters_{self.best_epoch}.png"),
            #     n_neighbors=50, min_dist=0.05, target_weight=0.9,   # 更“两个大簇”
            #     point_size=9, alpha=0.85
            # )
            print(f"[t-SNE] figure saved to: {tsne_png}")

            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss

