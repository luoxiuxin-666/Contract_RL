# 引入我们刚才写的类 和 你的画图函数
from data_manager import ExperimentDataManager
from plot_picture_2 import plot_learning_curves_  # 假设你的画图函数在这个文件里
from plot_picture import plot_ic_verification


def compare_pic():
    global data_manager
    # 1. 实例化管理器
    data_manager = ExperimentDataManager(save_dir="results/my_experiments")
    # 2. 读取数据
    loaded_results = data_manager.load_metrics("diff_compare")
    # 3. 按照原来的格式提取数据
    total_data = loaded_results['total_data']
    total_uti = loaded_results['total_uti']
    # 4. 调用你之前修改好的画图函数重新生成图片
    # 画 total_data 的图 (联合展示)
    plot_learning_curves_(
        metrics_dict=total_data,
        current_episode=3000,
        picture_name='total_data',
        window_size=20,
        combine_plots=True
    )
    # 画 total_uti 的图 (独立展示)
    plot_learning_curves_(
        metrics_dict=total_uti,
        current_episode=3000,
        picture_name='total_uti',
        window_size=5,
        combine_plots=True
    )


def ic_verification(env, action_dict, node_type='DN'):
    plot_ic_verification(env, action_dict, node_type=node_type)


if __name__ == '__main__':
    from Contract_Env import Contract_Environment
    from Contract_Config import Config
    cfg = Config()
    env = Contract_Environment(cfg)
    compare_pic()

    cn_contract = []
    dn_contract = []
