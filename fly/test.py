

width = 8
height = 8

class Material(object):
    colour = (255,255,255)  # 颜色
    transparent = 0 #透明度，[0,1],越大越透明
    solidity = 0.5  # 硬度 [0,1]
    attribute = None  # 阴阳(冷热、刚柔) 五行 属性
    energy = None #能量
    energy_output_rate = 1 #能量输出转换率，默认100%
    energy_input_rate = 1  #能量输入转换率，默认100%


class Plant(object):
    position = (0,0) #位置
    shape = None #形状
    material = Material() #材质
    talent = None #天赋









