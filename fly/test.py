
width = 8
height = 8
class Material(object):#材质
    attribute = None # 气质，光质，水质，石质，土质，木质，肉质
    colour = (255,255,255)  # 颜色
    transparent = 0 #透明度，[0,1],越大越透明
    solidity = 0.5  # 硬度 [0,1]
    flexibility = 0.5 #柔性 [0,1]
    smoothness = 0.5 #光滑度 [0,1]
    energy = 0 #能量 [0,1]
    corrosive = 0 #腐蚀性 [0,1]
    toxicity = 0 #毒性 [0,1]

    energy_output_rate = 1 #能量输出转换率，[0,1]
    energy_input_rate = 1  #能量输入转换率，[0,1]
    talent = []  # 禀赋

class Stuff(object):#原料
    sub = [] #附属：材质, 原料
    position = [(0, 0)]  # 位置
    # shape = []  # 形状: 圆形，三角形，四方形

class Plant(object):
    material = Stuff() #原料
    talent = [] #天赋









