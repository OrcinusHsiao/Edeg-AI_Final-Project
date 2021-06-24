from machine import Pin, I2C,ADC    # 導入 Pin, I2C相關函式庫
from ssd1306 import SSD1306_I2C # 導入 SSD1306 I2C介面OLED函式庫
import aht10
import framebuf # 導入影格緩衝區函式庫
import utime # 導入時間相關函式庫
import sys # 導入系統標準輸出入相關函式庫
import select # 導入異步選擇函式庫

i2c = I2C(0, scl=Pin(17), sda=Pin(16), freq=400000)     # I2C#1初始化,設定SCL,SDA腳位及傳輸頻率       # Init I2C using pins GP8 & GP9 (default I2C0 pins)
print("I2C Address      : "+hex(i2c.scan()[0]).upper()) # 顯示OLED I2C起始位址
print("I2C Configuration: "+str(i2c))

# 顯示I2C組態
#設定SSD1306 OLED 驅動IC相關參數 
WIDTH  = 128 # 顯示寬度像素
HEIGHT = 64  # 顯示高度像素
oledi2c = I2C(1, scl=Pin(27), sda=Pin(26), freq=400000)
print("oledi2c Address      : "+hex(oledi2c.scan()[0]).upper()) # 顯示OLED I2C起始位址
print("oledi2c Configuration: "+str(oledi2c))
oled = SSD1306_I2C(WIDTH, HEIGHT, oledi2c) # SSD1306 OLED初始化
oled.fill(0) # 清除OLED顯示區(全部填零)

aht = aht10.AHT10(i2c)

adc = ADC(28)
count = (65535-10000)/100
cmd =""
predict = 0.0
predic_min=0
time_start = utime.ticks_ms() # 啟動ms計時器
while True:
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:        
        s = sys.stdin.readline().strip() #從標準輸人讀取一行文字並去除換行符號
        cmd = str(s)# 將收到的內容轉成字串
        #predict=cmd[0]
        #predic_min=cmd[1]
    
    mois = (65535-adc.read_u16())/count
    oled.fill(0)
    oled.text("Hum: %.2f %%"%aht.humidity(),0,0) # 在座標(0,0)位置顯示字串
    oled.text("Temp: %.2f C"%aht.temperature(),0,10) # 在座標(0,0)位置顯示字串
    oled.text("Mois: %.2f %%"%mois,0,20) # 在座標(0,0)位置顯示字串
    #oled.text(cmd,0,30)
    if(len(cmd)>0):
        tmp1 = float(cmd.split(',')[0])
        tmp2 = int(cmd.split(',')[1])
        oled.text("24hr_later: %.2f %%"%tmp1,0,40)
        oled.text("Dry: %dMin"%tmp2,0,50)
    oled.show()
    
    time_stop = utime.ticks_ms() # 取得目前時間
    if utime.ticks_diff(time_stop,time_start) > 2000: # 若先前時間比較大於0.5秒
        print(str(mois)+","+str(aht.temperature()))
        time_start = time_stop # 更新目前時間到先前時間
