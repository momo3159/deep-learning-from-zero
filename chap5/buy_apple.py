apple = 100
apple_count = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

price_without_tax = mul_apple_layer.forward(apple, apple_count)
price = mul_tax_layer.forward(price_without_tax, tax)
print(price)

d_price = 1
d_price_without_tax, d_tax = mul_tax_layer.backward(d_price)
d_apple, d_count = mul_apple_layer.backward(d_price_without_tax)
print(d_apple, d_count, d_tax)