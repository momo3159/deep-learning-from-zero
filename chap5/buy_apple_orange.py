from layer_naive import MulLayer, AddLayer

# input
apple_count = 2
apple_price = 100

orange_count = 3
orange_price = 150

tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
price_without_tax_layer = AddLayer()
price_layer = MulLayer()

apple_sum_without_tax = mul_apple_layer.forward(apple_count, apple_price)
orange_sum_without_tax = mul_orange_layer.forward(orange_count, orange_price)
price_without_tax = price_without_tax_layer.forward(apple_sum_without_tax, orange_sum_without_tax)
price = price_layer.forward(price_without_tax, tax)

d_price = 1
d_price_without_tax, d_tax = price_layer.backward(d_price)
d_apple_sum_without_tax, d_orange_sum_without_tax = price_without_tax_layer.backward(d_price_without_tax)
d_apple_count, d_apple = mul_apple_layer.backward(d_apple_sum_without_tax)
d_orange_count, d_orange = mul_orange_layer.backward(d_orange_sum_without_tax)

print(price)
print(d_tax, d_apple_count, d_apple, d_orange_count, d_orange)