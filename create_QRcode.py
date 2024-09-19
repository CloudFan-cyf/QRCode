from pyzbar.pyzbar import create_qrcode  
  
# 要生成二维码的字符串  
data = "Hello, world!"  
  
# 生成二维码图片并保存到本地  
qrcode = create_qrcode(data, "qrcode.png")