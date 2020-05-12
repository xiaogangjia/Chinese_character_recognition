import os

classes = ['良', '司', '林', '深', '合', '思', '光', '永', '起', '父', '流', '东', '春', '申', '玉', '室', '夜', '秋', '衣',
           '字', '命', '利', '清', '氏', '通', '世', '官', '章', '皇', '印', '右', '武', '建', '孝', '用', '从', '遂', '朝',
           '意', '亭', '宗', '立', '家', '来', '游', '西', '鼎', '臣', '我', '寒', '奉', '和', '名', '士', '受', '益', '左',
           '堂', '且', '常', '作', '黄', '物', '高', '御', '足', '令', '陵', '元', '始', '去', '易', '李', '虎', '使', '新',
           '老', '福', '九', '正', '侯', '今', '海', '唯', '敬', '徐', '莫', '必', '定', '女', '身', '重', '好', '宜', '守',
           '白', '雨', '更', '多', '周']

data_dir = 'dataset/train/'

cls_names = os.listdir(data_dir)


def data():
    trains = {}
    for cls in cls_names:
        img_name = os.listdir(data_dir + cls)
        #print(cls)
        #print(len(img_name))
        for img in img_name:
            i = img.split('.')[0]

            trains[i] = classes.index(cls)
    return trains
