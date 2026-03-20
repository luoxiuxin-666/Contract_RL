import os
import pickle


class ExperimentDataManager:
    """
    实验数据管理类：用于将训练过程中的字典、列表、NumPy数组等数据持久化保存到本地，
    以便后续随时加载并重新绘图，无需重新训练。
    """

    def __init__(self, save_dir="results/saved_metrics"):
        """
        初始化数据管理器
        :param save_dir: 数据保存的根目录，默认存放在 results/saved_metrics 下
        """
        self.save_dir = save_dir
        # 自动创建目录（如果不存在）
        os.makedirs(self.save_dir, exist_ok=True)

    def save_metrics(self, filename, **kwargs):
        """
        保存数据接口
        :param filename: 保存的文件名 (例如: 'experiment_1.pkl')
        :param kwargs: 任意数量的关键字参数 (例如: total_data=my_data, total_uti=my_uti)
        """
        # 自动补全后缀名
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        file_path = os.path.join(self.save_dir, filename)

        # 将传入的所有关键字参数打包成字典并保存
        with open(file_path, 'wb') as f:
            pickle.dump(kwargs, f)

        print(f"[DataManager] 成功保存数据至 -> {file_path}")
        print(f"[DataManager] 已保存的数据键值: {list(kwargs.keys())}")

    def load_metrics(self, filename):
        """
        读取数据接口
        :param filename: 要读取的文件名
        :return: 包含所有保存数据的字典
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        file_path = os.path.join(self.save_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[DataManager] 找不到文件: {file_path}")

        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        print(f"[DataManager] 成功从 {file_path} 加载数据")
        print(f"[DataManager] 可用的数据键值: {list(loaded_data.keys())}")
        return loaded_data