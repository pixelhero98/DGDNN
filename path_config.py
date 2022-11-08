

def path():
    return '/content/drive/MyDrive/Zinuo_Project/1_Stock_Recommendation/Colab_Notebooks/MY_Stock_Data'

def target_comps_path():
    return '/content/drive/MyDrive/Zinuo_Project/1_Stock_Recommendation/Colab_Notebooks/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv'

def dir_path():
    return '/content/drive/MyDrive/Zinuo_Project/1_Stock_Recommendation/'

def dataset_seg(index):
    if index == 'train':
        return 2013, 2015
    elif index == 'val':
        return 2014, 2016
    elif index == 'test':
        return 2015, 2017