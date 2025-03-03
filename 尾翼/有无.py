import os

# 定义后视图分割文件夹的路径
folder_path = "后视图分割"

# 定义之前后视图和之后后视图的子文件夹名称
before_folder = os.path.join(folder_path, "之前后视图")
after_folder = os.path.join(folder_path, "之后后视图")

# 检查两个子文件夹是否存在
if not os.path.exists(before_folder) or not os.path.exists(after_folder):
    print("之前后视图或之后后视图文件夹不存在，请检查路径。")
else:
    # 查找之前后视图文件夹中是否存在包含尾翼的.png文件
    before_has_tail = False
    for root, dirs, files in os.walk(before_folder):
        for file in files:
            if file.endswith('.png') and '尾翼' in file:
                before_has_tail = True
                break
        if before_has_tail:
            break

    # 查找之后后视图文件夹中是否存在包含尾翼的.png文件
    after_has_tail = False
    for root, dirs, files in os.walk(after_folder):
        for file in files:
            if file.endswith('.png') and '尾翼' in file:
                after_has_tail = True
                break
        if after_has_tail:
            break

    # 根据规则进行判断
    if (before_has_tail and after_has_tail) or (not before_has_tail and not after_has_tail):
        print("没问题")
    else:
        print("改装了")
