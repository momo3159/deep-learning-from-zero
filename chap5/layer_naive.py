
class MulLayer:
  def __init__(self):
    # backwardの処理で入力値が必要なので
    # 初期値をNoneにするのは、layerの定義と順伝播の処理を分けるため
      # mul_apple_layer = MulLayer(apple_count, apple_price)
      # mul_orange_layer = MulLayer(orange_count, orange_price)
      # price_without_tax_layer = AddLayer()
      # price_layer = MulLayer() <- これは前の結果が必要
    self.x = None
    self.y = None
  
  def forward(self, x, y):
    self.x = x
    self.y = y 
    
    return x * y
  
  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy

class AddLayer:
  def __init__(self):
    pass 
  
  def forward(self, x, y):
    return x + y 
  
  def backward(self, dout): 
    dx = dout  
    dy = dout 
  
    return dx, dy

