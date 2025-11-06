import pandas as pd
import numpy as np
import warnings
import pickle
import re
# 导入特征选择相关工具：F检验、互信息、FPR选择器、方差阈值
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectFpr, VarianceThreshold
# 导入数据标准化工具
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
# 导入重采样工具（用于bootstrap抽样）
from sklearn.utils import resample
# 导入自定义工具函数（假设用于获取新列、加载/写入数据、检查文件名）
from base_utils import get_new_cols, load_data, write_file, check_filename


def remove_rarely_mutated_genes(g, f=0.01):
    """
    移除突变频率过低的基因特征（过滤信息量少的突变特征）
    
    参数:
        g: 数据容器对象，需包含以下属性：
            - tcga_x_train: TCGA训练集数据（DataFrame）
            - cols['mut']: 突变特征列名列表
            - tcga_x_val, tcga_x_test: TCGA验证集、测试集
            - glass_x_train, glass_x_val, glass_x_test: GLASS数据集的训练/验证/测试集
        f: 突变频率阈值，低于此值的基因将被移除（默认1%）
    """
    # 计算每个突变基因在训练集中的突变频率（>0表示发生突变）
    mutation_freq = (g.tcga_x_train[g.cols['mut']] > 0).mean()
    # 筛选出频率≥阈值的基因
    keep_genes = mutation_freq[mutation_freq >= f].index.tolist()
    # 打印过滤信息
    print(f"{len(g.cols['mut'])-len(keep_genes)}个突变特征被移除（突变频率<{round(f*100)}%），剩余{len(keep_genes)}个突变特征")
    # 更新突变特征列名列表
    g.cols['mut'] = keep_genes
    # 整合所有需保留的特征列（临床+表达+突变）
    all_cols = [*g.cols['clin'], *g.cols['expr'], *g.cols['mut']]
    # 过滤所有数据集，仅保留筛选后的特征
    g.tcga_x_train, g.tcga_x_val, g.tcga_x_test = [
        df[all_cols].copy() for df in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]
    ]
    g.glass_x_train, g.glass_x_val, g.glass_x_test = [
        df[all_cols].copy() for df in [g.glass_x_train, g.glass_x_val, g.glass_x_test]
    ]


def log1p_mutations(g):
    """
    对突变特征应用log1p转换（log(1+x)），用于处理偏态分布数据，降低极端值影响
    
    参数:
        g: 数据容器对象，需包含突变特征列名（cols['mut']）及各数据集（tcga/glass的 train/val/test）
    """
    # 获取突变特征列名
    cols = g.cols['mut']
    # 遍历所有数据集（TCGA和GLASS的训练/验证/测试集）
    for attr in ['tcga_x_train', 'tcga_x_val', 'tcga_x_test', 
                 'glass_x_train', 'glass_x_val', 'glass_x_test']:
        # 获取当前数据集
        df = getattr(g, attr)
        # 对突变特征进行log1p转换（先转为float避免整数溢出）
        float_block = df[cols].astype(float).apply(np.log1p)
        # 更新数据集中的突变特征列
        df.loc[:, cols] = float_block


def remove_nearly_constant(g, threshold=1e-8):
    """
    移除方差接近0的近常数特征（此类特征信息量极低）
    
    参数:
        g: 数据容器对象，包含各数据集
        threshold: 方差阈值，低于此值的特征被移除（默认1e-8）
    """
    # 初始化方差阈值选择器，输出格式为DataFrame
    vt = VarianceThreshold(threshold=threshold).set_output(transform='pandas')
    # 用训练集拟合选择器并过滤训练集
    g.tcga_x_train = vt.fit_transform(g.tcga_x_train)
    # 用训练集拟合的选择器过滤其他数据集（保证过滤规则一致）
    g.tcga_x_val = vt.transform(g.tcga_x_val)
    g.tcga_x_test = vt.transform(g.tcga_x_test)
    g.glass_x_train = vt.transform(g.glass_x_train)
    g.glass_x_val = vt.transform(g.glass_x_val)
    g.glass_x_test = vt.transform(g.glass_x_test)


def apply_scaler(g, ctype='robust', etype='robust', mtype='robust'):
    """
    对不同类型的特征应用标准化/归一化处理（消除量纲影响）
    
    参数:
        g: 数据容器对象，包含各特征列及数据集
        ctype: 临床特征（如年龄）的标准化方法（None表示不处理）
        etype: 表达特征的标准化方法
        mtype: 突变特征的标准化方法
        支持的方法：'minmax'（MinMaxScaler）、'robust'（RobustScaler）、
                  'maxabs'（MaxAbsScaler）、'standard'（StandardScaler）
    """
    # 遍历特征类型（临床、表达、突变）及对应的列名和标准化方法
    for stype, cols in zip(
        [ctype, etype, mtype],  # 标准化方法列表
        [['age'], g.cols['expr'], g.cols['mut']]  # 对应特征列名
    ):
        # 若方法为None，则跳过该类型特征
        if stype is None:
            continue
        # 根据方法名选择对应的scaler
        scaler_class = {
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'maxabs': MaxAbsScaler,
            'standard': StandardScaler
        }[stype]
        tcga_sc = scaler_class()
        # 用TCGA训练集拟合scaler并转换训练集
        g.tcga_x_train[cols] = tcga_sc.fit_transform(g.tcga_x_train[cols])
        # 用训练集拟合的scaler转换其他数据集（保证尺度一致）
        g.tcga_x_val[cols] = tcga_sc.transform(g.tcga_x_val[cols])
        g.tcga_x_test[cols] = tcga_sc.transform(g.tcga_x_test[cols])
        g.glass_x_train[cols] = tcga_sc.transform(g.glass_x_train[cols])
        g.glass_x_val[cols] = tcga_sc.transform(g.glass_x_val[cols])
        g.glass_x_test[cols] = tcga_sc.transform(g.glass_x_test[cols])


class COMBAT:
    """
    实现简化版ComBat批处理效应校正（用于消除不同批次数据间的系统差异）
    注：标准ComBat同时校正均值和方差，此处为简化版仅校正均值
    """
    
    def __init__(self, g):
        """
        初始化批处理标签（如测序平台、批次ID等）并验证数据一致性
        
        参数:
            g: 数据容器对象，需包含：
                - tcga_tss, glass_tss: TCGA和GLASS数据的批处理标签（Series，索引为样本ID）
                - 各数据集（tcga/glass的train/val/test，索引为样本ID）
        """
        # 提取TCGA各数据集对应的批处理标签（保证样本索引一致）
        self.tcga_tss_train, self.tcga_tss_val, self.tcga_tss_test = [
            g.tcga_tss.loc[split.index].copy() 
            for split in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]
        ]
        # 提取GLASS各数据集对应的批处理标签
        self.glass_tss_train, self.glass_tss_val, self.glass_tss_test = [
            g.glass_tss.loc[split.index].copy() 
            for split in [g.glass_x_train, g.glass_x_val, g.glass_x_test]
        ]
        # 验证所有数据集的样本索引与批处理标签索引一致（避免校正错误）
        for split in ['train', 'val', 'test']:
            for ds in ['tcga', 'glass']:
                # 获取当前数据集
                x = getattr(g, f"{ds}_x_{split}")
                # 获取当前数据集对应的批处理标签
                tss = g.tcga_tss.loc[x.index].copy() if ds == 'tcga' else g.glass_tss.loc[x.index].copy()
                # 断言索引一致，否则抛出错误
                assert x.index.equals(tss.index), f"{ds.upper()} {split}数据集的批处理标签与特征数据索引不匹配"
    
    def correct_modality(self, g, mod):
        """
        对指定模态的特征（如表达或突变）应用批处理校正
        
        参数:
            g: 数据容器对象（需更新校正后的数据）
            mod: 模态名称，需在g.cols中存在（如'expr'表示表达特征，'mut'表示突变特征）
        """
        # 获取当前模态的特征列名
        mod_cols = g.cols[mod]
        
        # 步骤1：从训练集估计批次特异性均值和全局均值（校正基准）
        # TCGA数据的批次均值和全局均值
        tcga_batch_means, tcga_global_mean = self.estimate_combat_means(
            g.tcga_x_train[mod_cols], self.tcga_tss_train
        )
        # GLASS数据的批次均值和全局均值
        glass_batch_means, glass_global_mean = self.estimate_combat_means(
            g.glass_x_train[mod_cols], self.glass_tss_train
        )
        
        # 步骤2：应用校正（对所有数据集）
        # 校正TCGA数据集
        g.tcga_x_train.loc[:, mod_cols] = self.apply_combat_correction(
            g.tcga_x_train[mod_cols], self.tcga_tss_train, tcga_batch_means, tcga_global_mean
        )
        g.tcga_x_val.loc[:, mod_cols] = self.apply_combat_correction(
            g.tcga_x_val[mod_cols], self.tcga_tss_val, tcga_batch_means, tcga_global_mean
        )
        g.tcga_x_test.loc[:, mod_cols] = self.apply_combat_correction(
            g.tcga_x_test[mod_cols], self.tcga_tss_test, tcga_batch_means, tcga_global_mean
        )
        # 校正GLASS数据集
        g.glass_x_train.loc[:, mod_cols] = self.apply_combat_correction(
            g.glass_x_train[mod_cols], self.glass_tss_train, glass_batch_means, glass_global_mean
        )
        g.glass_x_val.loc[:, mod_cols] = self.apply_combat_correction(
            g.glass_x_val[mod_cols], self.glass_tss_val, glass_batch_means, glass_global_mean
        )
        g.glass_x_test.loc[:, mod_cols] = self.apply_combat_correction(
            g.glass_x_test[mod_cols], self.glass_tss_test, glass_batch_means, glass_global_mean
        )
    
    def estimate_combat_means(self, x_train_mod, batch_labels):
        """
        从训练集估计每个批次的均值和全局均值（校正的核心参数）
        
        参数:
            x_train_mod: 训练集中当前模态的特征数据（DataFrame）
            batch_labels: 训练集的批处理标签（Series）
        
        返回:
            batch_means: 字典，键为批次ID，值为该批次的特征均值（Series）
            global_mean: 全局均值（所有样本的特征均值，Series）
        """
        # 获取所有唯一批次
        batches = batch_labels.unique()
        batch_means = {}
        # 计算每个批次的特征均值
        for b in batches:
            # 筛选当前批次的样本
            idx = batch_labels == b
            batch_data = x_train_mod.loc[idx].copy()
            # 计算该批次各特征的均值（按列求均值）
            batch_means[b] = batch_data.mean(axis=0)
        # 计算全局均值（所有训练样本的特征均值）
        global_mean = x_train_mod.mean(axis=0)
        return batch_means, global_mean
    
    def apply_combat_correction(self, mod_df, batches, batch_means, global_mean):
        """
        应用批处理校正：样本值 = 原始值 - 批次均值 + 全局均值（消除批次偏移）
        
        参数:
            mod_df: 待校正的特征数据（DataFrame）
            batches: 样本的批处理标签（Series，索引与mod_df一致）
            batch_means: 训练集估计的批次均值（字典）
            global_mean: 训练集估计的全局均值（Series）
        
        返回:
            corrected: 校正后的特征数据（DataFrame）
        """
        # 初始化校正后的数据框
        corrected = pd.DataFrame(
            index=mod_df.index, 
            columns=mod_df.columns, 
            dtype=float
        )
        # 逐样本校正
        for i in mod_df.index:
            # 获取当前样本的批次
            b = batches.loc[i]
            # 若该批次在训练集中存在（有对应的批次均值）
            if b in batch_means:
                # 应用校正公式
                corrected.loc[i, :] = mod_df.loc[i] - batch_means[b] + global_mean
            else:
                # 若批次未在训练集中出现（如测试集新批次），仅做全局中心化（无批次校正）
                corrected.loc[i, :] = mod_df.loc[i] - global_mean + global_mean
                # 发出警告
                warnings.warn(f"校正数据中存在训练集未见过的批次'{b}'，跳过该批次的校正")
        return corrected.astype(float)


class CorrelationAnalysis:
    """
    分析特征间的相关性并移除高相关特征（减少冗余信息，避免多重共线性）
    支持基于已知基因优先级、方差、相关计数等策略选择保留特征
    """
    
    def __init__(self, action, glioma, corr_mx_fn, data_path='data/', threshold=0.95, 
                 calculate_pairs=True, corr_pairs_fn='', load_corr_pairs=False):
        """
        初始化相关性分析：计算或加载相关矩阵，识别高相关特征对
        
        参数:
            action: 操作类型，'load'表示加载已有相关矩阵，否则计算新矩阵
            glioma: 数据容器对象（需包含TCGA训练集及特征列）
            corr_mx_fn: 相关矩阵的文件名
            data_path: 数据存储路径（默认'data/'）
            threshold: 高相关阈值，相关系数≥此值的视为高相关（默认0.95）
            calculate_pairs: 是否计算高相关特征对（True则计算）
            corr_pairs_fn: 高相关特征对的文件名（用于保存/加载）
            load_corr_pairs: 是否加载已有高相关特征对（True则加载）
        """
        self.data_path = data_path
        # 获取相关矩阵（计算或加载）
        self.corr_mx = self._get_corr_mx(action, glioma, corr_mx_fn)
        # 计算或加载高相关特征对
        if calculate_pairs: 
            self.corr_pairs = self._get_corr_pairs(threshold, corr_pairs_fn, load_corr_pairs)
            print(f"识别到{len(self.corr_pairs)}对高相关特征")
        # 加载文献中已知的与胶质瘤相关的基因（用于优先级排序）
        self.known_genes = self._get_known_genes_from_literature()
    
    def remove_correlated_genes(self, action, glioma, fn='', primary_method='variance', 
                                secondary_method='corr count', prioritize_known_genes=True, 
                                get_removal_set_only=False, clin_features_to_drop=[], save_removal=True):
        """
        移除高相关特征，支持基于已知基因优先级和特征指标（方差、相关计数等）选择保留特征
        
        参数:
            action: 操作类型，'load'表示加载已有移除列表，否则计算新列表
            glioma: 数据容器对象（需更新特征列和数据集）
            fn: 移除列表的文件名（用于保存/加载）
            primary_method: 主要筛选方法（'variance'：保留高方差；'target_corr'：保留与目标高相关）
            secondary_method: 次要筛选方法（当主要方法无法区分时使用）
            prioritize_known_genes: 是否优先保留已知基因（True则移除与已知基因相关的未知基因）
            get_removal_set_only: 是否仅计算移除列表而不应用到数据（True则不更新数据）
            clin_features_to_drop: 需强制移除的临床特征列表
            save_removal: 是否保存移除列表（True则保存）
        """
        # 若为加载模式，直接加载移除列表并应用
        if action == 'load':
            with open(self.data_path + fn, 'rb') as f:
                self.remove = pickle.load(f)
            self._apply_removal(glioma)
            return
        
        # 初始化移除列表（分两步：已知基因优先级筛选 + 高相关对筛选）
        self.remove = {}
        # 步骤1：优先保留已知基因（移除与其高相关的未知基因）
        if prioritize_known_genes: 
            self._prioritize_known_genes(glioma)
        else: 
            # 若不优先已知基因，直接使用所有高相关对
            self.corr_pairs_step1 = self.corr_pairs.copy()
            self.remove['step 1'] = {'clin': [], 'expr': [], 'mut': []}
        
        # 统计主要/次要方法的使用次数
        self.counts = {'primary': 0, 'secondary': 0}
        # 存储待移除的特征
        to_remove = set()
        # 遍历高相关特征对（经步骤1筛选后）
        for _, row in self.corr_pairs_step1.iterrows():
            var1, var2 = row['var1'], row['var2']
            # 若其中一个已被标记移除，则跳过
            if var1 in to_remove or var2 in to_remove: 
                continue
            # 选择需移除的特征（基于主要/次要方法）
            drop_feature = self._select_drop_feature(
                glioma, var1, var2, primary_method, secondary_method
            )
            to_remove.add(drop_feature)
        
        # 步骤2：整理各类型特征的移除列表
        # 从待移除集合中提取表达特征
        expr_to_remove = get_new_cols(to_remove, glioma.cols['expr'])
        # 提取突变特征
        mut_to_remove = get_new_cols(to_remove, glioma.cols['mut'])
        self.remove['step 2'] = {
            'clin': clin_features_to_drop,  # 临床特征移除列表（用户指定）
            'expr': expr_to_remove,
            'mut': mut_to_remove
        }
        
        # 保存移除列表
        if save_removal and fn != '':
            with open(self.data_path + fn, 'wb') as f:
                pickle.dump(self.remove, f)
        
        # 应用移除列表（更新数据）
        if not get_removal_set_only:
            self._apply_removal(glioma)
            return
    
    def _apply_removal(self, glioma):
        """将移除列表应用到数据，更新特征列和所有数据集"""
        # 保存移除前的特征列（用于日志）
        glioma.cols_before_ca = {
            kind: glioma.cols[kind] for kind in ['clin', 'expr', 'mut']
        }
        # 更新特征列（保留未被移除的特征）
        glioma.cols = {
            kind: [
                f for f in glioma.cols_before_ca[kind] 
                if f not in self.remove['step 1'][kind] + self.remove['step 2'][kind]
            ]
            for kind in ['clin', 'expr', 'mut']
        }
        # 整合所有保留的特征
        self.keep = glioma.cols['clin'] + glioma.cols['expr'] + glioma.cols['mut']
        # 过滤所有数据集
        glioma.tcga_x_train = glioma.tcga_x_train[self.keep].copy()
        glioma.tcga_x_val = glioma.tcga_x_val[self.keep].copy()
        glioma.tcga_x_test = glioma.tcga_x_test[self.keep].copy()
        glioma.glass_x_train = glioma.glass_x_train[self.keep].copy()
        glioma.glass_x_val = glioma.glass_x_val[self.keep].copy()
        glioma.glass_x_test = glioma.glass_x_test[self.keep].copy()
        # 打印过滤结果
        print(f"移除的特征数：临床特征{len(glioma.cols_before_ca['clin']) - len(glioma.cols['clin'])}个，"
              f"表达特征{len(glioma.cols_before_ca['expr']) - len(glioma.cols['expr'])}个，"
              f"突变特征{len(glioma.cols_before_ca['mut']) - len(glioma.cols['mut'])}个")
        print(f"剩余的特征数：临床特征{len(glioma.cols['clin'])}个，"
              f"表达特征{len(glioma.cols['expr'])}个，"
              f"突变特征{len(glioma.cols['mut'])}个\n")
    
    def _get_corr_mx(self, action, glioma, corr_mx_fn):
        """计算或加载特征相关矩阵（仅基于TCGA训练集，避免数据泄露）"""
        if action == 'load':
            # 加载已有相关矩阵
            return load_data('pd', corr_mx_fn, self.data_path)
        # 计算表达和突变特征的相关矩阵（皮尔逊相关系数）
        corr_mx = glioma.tcga_x_train[glioma.cols['expr'] + glioma.cols['mut']].corr()
        # 保存相关矩阵
        corr_mx.to_pickle(self.data_path + check_filename(corr_mx_fn, 'pkl'))
        return corr_mx
    
    def _get_corr_pairs(self, threshold, corr_pairs_fn, load_corr_pairs):
        """识别高相关特征对（仅上三角矩阵，避免重复）"""
        if load_corr_pairs:
            # 加载已有高相关对
            return load_data('pd', corr_pairs_fn, self.data_path)
        # 提取相关矩阵的上三角部分（k=1表示排除对角线）并取绝对值
        uppertri = self.corr_mx.where(
            np.triu(np.ones(self.corr_mx.shape), k=1).astype(bool)
        ).abs()
        # 转换为长表格式（var1, var2, correlation）
        corr_pairs = uppertri.stack().reset_index()
        corr_pairs.columns = ['var1', 'var2', 'correlation']
        # 筛选出相关系数≥阈值的高相关对
        corr_pairs = corr_pairs.query("correlation >= @threshold")
        # 保存高相关对
        corr_pairs.to_pickle(self.data_path + check_filename(corr_pairs_fn, 'pkl'))
        return corr_pairs
    
    def _get_known_genes_from_literature(self):
        """从文献中获取已知的与胶质瘤相关的基因（用于优先级排序）"""
        # 文献来源：PMC9427889, PMC6407082, PMC2818769, PMC3910500, PMC3443254
        known_genes = {
            'IDH1', 'IDH2', 'BRAF', 'CDKN2A', 'CDKN2B', 'ATRX', 'TERT', 'TP53', 'EGFR', 
            'KIAA1549', 'MGMT', 'MYB', 'MYBL1', 'YAP1', 'RELA', 'MYCN', 'SMARCB1', 'NF1', 
            'NF2', 'MAPK', 'PIK3R1', 'PIK3CA', 'RB1', 'PTEN', 'PDGFRA', 'ERBB2', 'CHI3L1', 
            'MET', 'CD44', 'MERTK', 'NES', 'CDK4', 'CCND2', 'NOTCH3', 'JAG1', 'LFNG', 'SMO', 
            'GAS1', 'GLI2', 'TRADD', 'RELB', 'TNFRSF1A', 'NKX2-2', 'OLIG2', 'CDKN1A', 'DCX', 
            'DLL3', 'ASCL1', 'TCF4', 'NEFL', 'GABRA1', 'SYT1', 'SLC12A5', 'LZTR1', 'SPTA1', 
            'GABRA6', 'KEL', 'CDK6', 'MDM2', 'SOX2', 'CCND1', 'CCNE2', 'QKI', 'TGFBR2', 'CIC', 'FUBP1'
        }
        return sorted(known_genes)
    
    def _prioritize_known_genes(self, glioma):
        """
        优先保留已知基因：移除与已知基因高相关的未知基因
        若高相关对中两个均为已知基因，则保留到下一步处理
        """
        # 构建已知基因的正则表达式（匹配以已知基因开头的特征，如"IDH1_mut"）
        known_gene_pattern = f"^(?:{'|'.join(self.known_genes)})_"
        # 筛选包含至少一个已知基因的高相关对
        temp = self.corr_pairs[
            self.corr_pairs['var1'].str.contains(known_gene_pattern, case=False, na=False) |
            self.corr_pairs['var2'].str.contains(known_gene_pattern, case=False, na=False)
        ]
        
        # 标记需移除的未知基因
        to_remove = set()
        # 记录两个均为已知基因的相关对索引
        both_known_idxs = []
        for idx, row in temp.iterrows():
            # 判断var1是否为已知基因相关特征
            var1_known = bool(re.match(known_gene_pattern, row['var1']))
            # 判断var2是否为已知基因相关特征
            var2_known = bool(re.match(known_gene_pattern, row['var2']))
            
            if not var1_known:
                # var1是未知基因，移除
                to_remove.add(row['var1'])
            elif not var2_known:
                # var2是未知基因，移除
                to_remove.add(row['var2'])
            else:
                # 两者均为已知基因，保留该对到下一步
                both_known_idxs.append(idx)
        
        # 保存两个均为已知基因的相关对
        self.both_known = temp.loc[both_known_idxs]
        # 过滤高相关对：移除包含已标记移除特征的对
        self.corr_pairs_step1 = self.corr_pairs[
            ~self.corr_pairs['var1'].isin(to_remove) & 
            ~self.corr_pairs['var2'].isin(to_remove)
        ].copy()
        # 记录步骤1移除的特征
        self.remove['step 1'] = {
            'clin': [],  # 临床特征不参与此步骤
            'expr': get_new_cols(to_remove, glioma.cols['expr']),  # 表达特征中需移除的
            'mut': get_new_cols(to_remove, glioma.cols['mut'])     # 突变特征中需移除的
        }
        # 打印步骤1结果
        print(f"因与已知基因高相关而移除的基因数：{len(to_remove)}个 "
              f"（解决了{len(self.corr_pairs) - len(self.corr_pairs_step1)}对高相关）")
    
    def _select_drop_feature(self, glioma, f1, f2, primary_method='variance', secondary_method='corr count'):
        """
        从高相关对中选择需移除的特征（基于主要/次要方法）
        
        参数:
            glioma: 数据容器对象（含训练集数据）
            f1, f2: 高相关对中的两个特征
            primary_method: 主要筛选方法
            secondary_method: 次要筛选方法（主要方法无法区分时使用）
        
        返回:
            需移除的特征名称
        """
        # 计算特征指标（基于主要方法）
        primary = self._calculate_feature_metrics(glioma, f1, f2, method=primary_method)
        # 若两个特征的指标不同，基于主要方法选择
        if primary.loc[f1] != primary.loc[f2]: 
            self.counts['primary'] += 1
            # 对于方差/目标相关：移除指标低的；对于离群值/相关计数：移除指标高的
            return primary.idxmin() if primary_method in ['variance', 'target_corr'] else primary.idxmax()
        
        # 主要方法无法区分时，使用次要方法
        self.counts['secondary'] += 1
        secondary = self._calculate_feature_metrics(glioma, f1, f2, method=secondary_method)
        return secondary.idxmin() if secondary_method in ['variance', 'target_corr'] else secondary.idxmax()
    
    def _calculate_feature_metrics(self, glioma, f1, f2, method):
        """计算特征的指标（用于筛选）"""
        if method == 'variance':
            # 特征方差（高方差特征更可能保留）
            return glioma.tcga_x_train[[f1, f2]].var()
        if method == 'target_corr':
            # 与目标变量的相关性（绝对值，高相关更可能保留）
            return glioma.tcga_x_train[[f1, f2]].corrwith(glioma.tcga_y_train).abs()
        if method == 'outliers':
            # 离群值数量（低离群值更可能保留）
            return pd.Series(data={
                f1: self._count_outliers(glioma.tcga_x_train[f1]),
                f2: self._count_outliers(glioma.tcga_x_train[f2])
            })
        if method == 'corr count':
            # 高相关对数量（参与高相关对少的更可能保留）
            return pd.Series(data={
                f1: len(self.corr_pairs.query("var1 == @f1 or var2 == @f1")),
                f2: len(self.corr_pairs.query("var1 == @f2 or var2 == @f2"))
            })
    
    def _count_outliers(self, s):
        """用IQR方法计算特征的离群值数量（IQR=Q3-Q1，离群值定义为<Q1-1.5*IQR或>Q3+1.5*IQR）"""
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum()


class ExpressionStabilitySelector:
    """
    基于稳定性选择（Stability Selection）筛选表达特征
    通过bootstrap抽样多次选择特征，保留在多数抽样中被选中的稳定特征
    """
    
    def __init__(self, action, x=None, y=None, method='fpr', fn='', dp='data/', 
                 n_boots=100, fpr_alpha=0.05, mi_quantile=0.5, rs=42):             
        """
        初始化稳定性选择器：计算或加载特征选择频率
        
        参数:
            action: 操作类型，'load'表示加载已有选择频率，否则计算
            x: 特征数据（DataFrame，仅当action不为'load'时需提供）
            y: 目标变量（Series，仅当action不为'load'时需提供）
            method: 单次选择方法，'fpr'（基于F检验的FPR选择）或'mi'（基于互信息）
            fn: 选择频率的文件名（用于保存/加载）
            dp: 数据存储路径（默认'data/'）
            n_boots: bootstrap抽样次数（默认100）
            fpr_alpha: FPR选择的显著性水平（默认0.05）
            mi_quantile: 互信息选择的阈值分位数（默认0.5，即中位数）
            rs: 随机种子（保证可重复性）
        """
        # 加载已有选择频率或计算新频率
        self.sel_freq = load_data('pd', fn, 'data/') if action == 'load' else self.calc_freq(
            x, y, method, fn, n_boots, fpr_alpha, mi_quantile, rs
        )
    
    def calc_freq(self, x, y, method, fn, n_boots, fpr_alpha, mi_quantile, rs):  
        """计算特征的选择频率（在bootstrap抽样中被选中的比例）"""
        np.random.seed(rs)
        # 初始化特征计数（记录每个特征被选中的次数）
        feature_counts = pd.Series(0, index=x.columns)
        
        # 多次bootstrap抽样
        for i in range(n_boots):
            # 有放回抽样（保持类别分布）
            x_boot, y_boot = resample(
                x, y, 
                stratify=y,  # 分层抽样，保证目标变量分布与原数据一致
                n_samples=len(y),  # 抽样量与原数据相同
                replace=True,  # 有放回
                random_state=rs + i  # 每次抽样种子不同
            )
            
            # 基于当前抽样数据选择特征
            if method == 'fpr':
                # 基于F检验的FPR选择（控制假阳性率）
                selector = SelectFpr(score_func=f_classif, alpha=fpr_alpha)
                selector.fit(x_boot, y_boot)
                # 获取被选中的特征
                selected = x_boot.columns[selector.get_support()]
            elif method == 'mi':
                # 基于互信息（特征与目标的依赖关系）
                scores = mutual_info_classif(x_boot, y_boot, random_state=rs + i)
                # 动态阈值：当前抽样中互信息的分位数
                mi_thresh = np.quantile(scores, mi_quantile)
                # 选择互信息≥阈值的特征
                selected = x_boot.columns[np.array(scores) >= mi_thresh]
            
            # 更新特征计数
            feature_counts[selected] += 1
        
        # 计算选择频率（被选中次数 / 总抽样次数）
        selection_freq = feature_counts / n_boots
        # 保存选择频率
        if fn != '': 
            write_file('pd', fn, selection_freq, dp)
        return selection_freq
    
    def select_by_threshold(self, stability_threshold=0.8):
        """根据稳定性阈值选择特征（选择频率≥阈值的特征）"""
        self.selection = self.sel_freq[self.sel_freq >= stability_threshold].index.tolist()
        print(f"选择的表达特征数：{len(self.selection)}个")
    
    def apply_to_dataset(self, g):
        """将筛选后的表达特征应用到所有数据集"""
        # 更新特征列定义
        g.features = {
            'clin': g.cols['clin'], 
            'expr': self.selection, 
            'mut': g.cols['mut']
        }
        g.cols['expr'] = g.features['expr']
        # 整合所有需保留的特征
        all_cols = [*g.cols['clin'], *g.cols['expr'], *g.cols['mut']]
        # 过滤所有数据集
        g.tcga_x_train, g.tcga_x_val, g.tcga_x_test = [
            df[all_cols] for df in [g.tcga_x_train, g.tcga_x_val, g.tcga_x_test]
        ]
        g.glass_x_train, g.glass_x_val, g.glass_x_test = [
            df[all_cols] for df in [g.glass_x_train, g.glass_x_val, g.glass_x_test]
        ]
