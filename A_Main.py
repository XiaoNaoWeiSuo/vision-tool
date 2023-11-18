
import cv2 #摄像头控制，照片处理，机器视觉模块
from decimal import Decimal
import cmath
import math
#from threading import Thread#预备多线程
import sys
import time
from PyQt5 import  QtWidgets,QtGui,QtCore
from PyQt5.QtCore import QTimer
import numpy as np
import ast
from Ui_GUI import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtGui import QPixmap,QImage
#轮廓提取与处理  pat:原版图片地址  poly:拟合精度 fian:高斯模糊卷积参数  middle:二值中值    返回：拟合矩阵，重心坐标
def ContourD(pat,poly,fian,middle):
    image_F=cv2.imread(pat)
    image_M=cv2.blur(image_F,(int(fian),int(fian)))           #高斯模糊
    #cv2.imwrite(sav,image_M)
    #rtk,img=cv2.threshold(image_M,int(middle),255,cv2.THRESH_BINARY_INV)
    #cv2.imwrite(sav,image)
    gray = cv2.cvtColor(image_M,cv2.COLOR_BGR2GRAY)
    DUIBI=FileChange("DUIBI",None)
    if DUIBI=="cv2.THRESH_BINARY_INV":
        ret, binary = cv2.threshold(gray,int(middle),100,cv2.THRESH_BINARY_INV) #二值化，
    elif DUIBI=="cv2.THRESH_BINARY":
        ret, binary = cv2.threshold(gray,int(middle),100,cv2.THRESH_BINARY)
    cv2.imwrite("textTry.jpg",binary)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#   待测物体轮廓物体
#   提取数据
    data=[]
    tip=[]
    for i in range(len(contours)):
        cnt = contours[i]
        perimeter = cv2.arcLength(cnt,True)
        if perimeter>=100:
            data.append(round(perimeter,4))
            tip.append(i)
#   冒泡排序，筛选出第二大的轮廓
    for _ in range(len(data)):
        for num in range(len(data)-1):
            if data[num]>data[num+1]:
                lins=data[num]
                lint=tip[num]
                data[num]=data[num+1]
                tip[num]=tip[num+1]
                data[num+1]=lins
                tip[num+1]=lint

    #Douglas-Peucker算法
    try:
        cnt = contours[tip[-2]]
    except:
        cnt = contours[0]
    proce=[0]
    line_tipe=[]
    angle=cv2.approxPolyDP(cnt,int(poly),False)
    #获取顶角序数
    for b in range(len(angle)):
        for t in range(proce[-1],len(cnt)):
            if str(angle[b])==str(cnt[t]):
                proce.append(t)
                break
    #复现线性回归方程
    proce.pop(0)
    #边线简约点筛
    data=[]
    for p in range(len(proce)-1):
        data.append([])
        sor=proce[p]
        line_tipe.append(proce[p])
        Fpoint=proce[p]
        Spoint=proce[p+1]
        distin=(Spoint-Fpoint)
        if distin>=5:
            farItem=distin//5
        else:
            farItem=1
        for _ in range(5):
            a=str(cnt[sor][0][0])
            b=str(cnt[sor][0][1])
            data[p].append(cnt[sor][0].tolist())
            sor+=farItem
    bAngle=[]
    aAngle=[]
    P_Son=0
    P_Mom=0
    for h in range(len(proce)):
        npk=h+1
        P_Son=P_Mom=0
        if h==len(proce)-1:
            break
        else:
            X_Y_a=np.average(cnt[proce[h]:proce[npk]],axis=0)
            for tip in range(proce[h],proce[npk]):
                P_Son=P_Son+(int(cnt[tip][0][0])-int(X_Y_a[0][0]))*(int(cnt[tip][0][1])-int(X_Y_a[0][1]))
                P_Mom=P_Mom+(int(cnt[tip][0][0])-int(X_Y_a[0][0]))**2
        bAngle.append(round(P_Son/P_Mom,6))
        aAngle.append(round(X_Y_a[0][1]-bAngle[-1]*X_Y_a[0][0],6))
    FileChange("A_Angle",str(aAngle))
    FileChange("B_Angle",str(bAngle))
    #print("PROCE",proce,"A_Angle",aAngle,"B_Angle",bAngle)
    M = cv2.moments(cnt)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    bx=int(image_F.shape[1]/2)
    by=int(image_F.shape[0]/2)

    #储存数据

    #删除最后一个干扰元素:
    Nangle=np.delete(angle,-1,0)

    FileChange("angle",str(Nangle))
    FileChange("AX",cx)
    FileChange("AY",cy)
    FileChange("BX",bx)
    FileChange("BY",by)
    return Nangle,data

#   绘制拟合  anDta:拟合矩阵   phone:原图地址  cx,xy:物体重心
def Draw(anDta,phone,cx,cy,bx,by):
    Phimge=cv2.imread(phone)
#       绘制拟合边线
    Phimge_F=cv2.drawContours(Phimge,[anDta],-1,(0,255,0),2)
#       绘制原图  物体重心连接图片中心
    cv2.line(Phimge_F,(int(cx),int(cy)),(int(bx),int(by)),(0,0,255),5)
    cv2.line(Phimge_F,(int(cx),int(cy)),(anDta[0][0][0],anDta[0][0][1]),(0,255,0),8)#将第一个点和重心的连线变成绿色
    cv2.line(Phimge_F,(int(cx),int(cy)),(anDta[1][0][0],anDta[1][0][1]),(0,0,255),8)#将第二个点和中兴的连线变成红色
#       绘制采样点与物体重心连线
    for v in range(len(anDta)):
        cv2.line(Phimge_F,(int(cx),int(cy)),(int(anDta[v][0][0]),int(anDta[v][0][1])),(255,0,0),3)
    New="NewPicture.jpg"
    cv2.imwrite(New,Phimge_F)

#   选择    point和lines传入的都是含零序号
def Option(anDta,picture,point,lines,sign,nob):
    if sign=="2":#PL
        Phimge=cv2.imread(picture)
        cv2.line(Phimge,(0,0),(int(anDta[point][0][0]),int(anDta[point][0][1])),(87,139,46),8)
        line=lines
        cop_line=line+1
        if line>=len(anDta)-1:
            cop_line=0
            cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(0,69,255),9)
        else:
            cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(0,69,255),9)
    elif sign=="0":#pp
        Phimge=cv2.imread(picture)
        cv2.line(Phimge,(0,0),(int(anDta[point][0][0]),int(anDta[point][0][1])),(0,255,0),8)
        cv2.line(Phimge,(20,0),(int(anDta[lines][0][0]),int(anDta[lines][0][1])),(0,0,255),8)
    elif sign=="1":#LL
        Phimge=cv2.imread(picture)
        line=lines
        poin=point
        cop_line=line+1
        cop_poin=poin+1
        if nob=="F":
            if line>=len(anDta)-1:
                cop_line=0
                cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(87,139,46),9)
            else:
                cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(87,139,46),9)
            if poin>=len(anDta)-1:
                cop_poin=0
                cv2.line(Phimge,(int(anDta[poin][0][0]),int(anDta[poin][0][1])),(int(anDta[cop_poin][0][0]),int(anDta[cop_poin][0][1])),(0,69,255),7)
            else:
                cv2.line(Phimge,(int(anDta[poin][0][0]),int(anDta[poin][0][1])),(int(anDta[cop_poin][0][0]),int(anDta[cop_poin][0][1])),(0,69,255),7)
        elif nob=="S":
            if line>=len(anDta)-1:
                cop_line=0
                cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(0,69,255),7)
            else:
                cv2.line(Phimge,(int(anDta[line][0][0]),int(anDta[line][0][1])),(int(anDta[cop_line][0][0]),int(anDta[cop_line][0][1])),(0,69,255),7)
            if poin>=len(anDta)-1:
                cop_poin=0
                cv2.line(Phimge,(int(anDta[poin][0][0]),int(anDta[poin][0][1])),(int(anDta[cop_poin][0][0]),int(anDta[cop_poin][0][1])),(87,139,46),9)
            else:
                cv2.line(Phimge,(int(anDta[poin][0][0]),int(anDta[poin][0][1])),(int(anDta[cop_poin][0][0]),int(anDta[cop_poin][0][1])),(87,139,46),9)
    kape="OptionPicture.jpg"
    cv2.imwrite(kape,Phimge)

#边线函数计算  两点式 point:点位序号  line:直线序号     angle:近拟矩阵     nob:模式信号   dtat:采样点数据
def Calculate(point,line,angle,nob,dtat):
    aJ=FileChange("A_Angle",None)
    A_J=eval(aJ)
    bJ=FileChange("B_Angle",None)
    B_J=eval(bJ)
    if nob=="2":
        print("dian:",point,"xian:",line)
        a1=Decimal(int(angle[point][0][0])).quantize(Decimal("0.0000"))
        b1=Decimal(int(angle[point][0][1])).quantize(Decimal("0.0000"))
        print(a1,b1)
        print(A_J[line],B_J[line])
        Distin_son=Decimal(float(abs(B_J[line]*float(a1))-float(1*b1)+float(A_J[line]))).quantize(Decimal("0.0000"))
        Distin_mom=Decimal(cmath.sqrt((float(B_J[line])**2+float(1))).real).quantize(Decimal("0.0000"))
        Distin=round(Distin_son/Distin_mom,5)
        return Distin
    elif nob=="1":      #点线功能基本完成，预计添加采样点数量选择，以提高精度
        d_1=d_2=[]
        for npc in range(2):
            for xpc in range(5):
                if npc == 0:
                    d_1.append((B_J[line]*dtat[point][xpc][0]-1*dtat[point][xpc][1]+A_J[line])/math.sqrt(B_J[line]*B_J[line]+1).real)
                else:
                    d_2.append((B_J[point]*dtat[line][xpc][0]-1*dtat[line][xpc][1]+A_J[point])/math.sqrt(B_J[point]*B_J[point]+1).real)
        ter_1 = (d_1[0]+d_1[1]+d_1[2]+d_1[3]+d_1[4])/5
        ter_2 = (d_2[0]+d_2[1]+d_2[2]+d_2[3]+d_2[4])/5
        theend=(ter_1+ter_2)/2
        return theend
    elif nob=="0":
        x1=angle[line][0][0]
        y1=angle[line][0][1]
        x2=angle[point][0][0]
        y2=angle[point][0][1]
        kety=math.sqrt((y2-y1)**2+(x1-x2)**2)
        return kety
#   操作数据文件str1:数据名，str2:数据参数·······当str2为空时返回str1保存的参数
def FileChange(str1,str2):
    set_file=open("setting.txt","r+")
    set_file_dta=set_file.read()
    set_file.close()
    #print(set_file_dta)
    #print(type(set_file_dta))
    set_file_dta=eval(set_file_dta)
    if str2==None:
        return str(set_file_dta[str1])
    else:
        set_file_dta[str1]=str2
        file_data=open("setting.txt","w+")
        file_data.write(str(set_file_dta))
    set_file.close()

#   读取采样点并返回矩阵
def readAngle():
    text=FileChange("angle",None)
    text = text.replace(",", " ")
    text = str(text.replace('\n', ''))
    listDta=list(text)
    for i in range(1,len(listDta)-2):
        if listDta[i]=="]" and listDta[i+1]==" " :
            listDta[i+1]=","
        if listDta[i]==" " and listDta[i-1]!="[" and listDta[i-1]!=",":
            listDta[i]=","
        if listDta[i]=="," and listDta[i-1]==" ":
            listDta[i]=" "
    ending="".join(listDta)
    a = np.array(ast.literal_eval(ending))
    return a

class Login(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(Login,self).__init__(parent)
        self.setupUi(self)
        #配置Blurry
        self.Blurry_Slider.setMaximum(100)
        self.Blurry_Slider.setMinimum(0)
        npc=FileChange("Blurry",None)
        self.Blurry_Slider.setValue(int(npc))
        self.label_Blurry.setText(npc)
        self.Blurry_Slider.valueChanged.connect(lambda:self.chang(self.Blurry_Slider,self.label_Blurry))
        self.Blurry_Slider.sliderReleased.connect(lambda:self.lock("Blurry",self.Blurry_Slider))
        #配置Median
        self.Median_Slider.setMaximum(255)
        self.Median_Slider.setMinimum(0)
        npc=FileChange("Median",None)
        self.Median_Slider.setValue(int(npc))
        self.label_Median.setText(npc)
        self.Median_Slider.valueChanged.connect(lambda:self.chang(self.Median_Slider,self.label_Median))
        self.Median_Slider.sliderReleased.connect(lambda:self.lock("Median",self.Median_Slider))
        #配置Fit
        self.Fit_Slider.setMaximum(100)
        self.Fit_Slider.setMinimum(1)
        npc=FileChange("Fit",None)
        self.Fit_Slider.setValue(int(npc))
        self.label_Fit.setText(npc)
        self.Fit_Slider.valueChanged.connect(lambda:self.chang(self.Fit_Slider,self.label_Fit))
        self.Fit_Slider.sliderReleased.connect(lambda:self.lock("Fit",self.Fit_Slider))
        #摄像机
        self.init()
        #self.cap = cv2.VideoCapture(1)  # 初始化摄像头
        self.label_camera.setScaledContents(True)  # 图片自适应
        self.label_Phote.setScaledContents(True)  # 图片自适应

        #配置data
        npc=FileChange("data",None)
        self.lineEdit_data.setText(npc)
        self.pushButton_data.clicked.connect(self.cun)

        #开始
        self.pushButton_start.clicked.connect(self.StartDREW)
        #选择
        self.comboBox_duibi.currentIndexChanged.connect(self.duibi)
#       PL
        self.pushButton_lastPoint.clicked.connect(lambda:self.lastpoint(None))
        self.pushButton_nextPoint.clicked.connect(lambda:self.nextpoint(None))
        self.pushButton_lastLine.clicked.connect(lambda:self.lastline(None))
        self.pushButton_nextLine.clicked.connect(lambda:self.nextline(None))
#       PP
        self.pushButton_PP_F_lastpoint.clicked.connect(lambda:self.lastpoint("F"))
        self.pushButton_PP_F_nextpoint.clicked.connect(lambda:self.nextpoint("F"))
        self.pushButton_PP_S_lastpoint.clicked.connect(lambda:self.lastpoint("S"))
        self.pushButton_PP_S_nextpoint.clicked.connect(lambda:self.nextpoint("S"))
#       LL
        self.pushButton_LL_F_lastline.clicked.connect(lambda:self.lastline("F"))
        self.pushButton_LL_F_nextline.clicked.connect(lambda:self.nextline("F"))
        self.pushButton_LL_S_lastline.clicked.connect(lambda:self.lastline("S"))
        self.pushButton_LL_S_nextline.clicked.connect(lambda:self.nextline("S"))
#       计算
        self.pushButton_Caculater.clicked.connect(self.caculater)
        self.tabWidget.currentChanged.connect(self.Tabwidchange)

#       标签页信号反馈，存在启动时配置文件不同步的情况，bug待办
    def Tabwidchange(self):
        names=self.tabWidget.currentIndex()
        if str(names)=="0":
            self.label_5.setText("点：")
            self.label_8.setText("点：")
        elif str(names)=="1":
            self.label_8.setText("边线：")
            self.label_5.setText("边线：")
        elif str(names)=="2":
            self.label_5.setText("边线：")
            self.label_8.setText("点：")
        FileChange("morden",names)
    def init(self):
        # 定时器让其定时读取显示图片
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_image)
        # 打开摄像头
        self.pushButton_camera.clicked.connect(self.open_camera)

        # 拍照
        self.pushButton_take.clicked.connect(self.taking_pictures)

    def open_camera(self):
        self.cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # 摄像头
        self.camera_timer.start(30)  # 每20毫秒读取一次，即刷新率为50帧

    def show_image(self):
        flag, self.image = self.cap.read()  # 从视频流中读取图片
        image_show = cv2.resize(self.image, (2048,1536))  # 把读到的帧的大小重新设置为
        #image_show = self.image
        width, height = image_show.shape[:2]  # 行:宽，列:高
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)  # opencv读的通道是BGR,要转成RGB
        image_show = cv2.flip(image_show, 1)  # 水平翻转，因为摄像头拍的是镜像的。
        # 把读取到的视频数据变成QImage形式(图片数据、高、宽、RGB颜色空间，三个通道各有2**8=256种颜色)

        self.showImage = QtGui.QImage(image_show.data, height, width, QImage.Format_RGB888)
        self.label_camera.setPixmap(QPixmap.fromImage(self.showImage))  # 往显示视频的Label里显示QImage
        self.label_camera.setScaledContents(True) #图片自适应

    def taking_pictures(self):
        if self.cap.isOpened():
            FName = fr"images/cap{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
            print(FName)
            self.label_Phote.setPixmap(QtGui.QPixmap.fromImage(self.showImage))
            # self.showImage.save(FName + ".jpg", "JPG", 100)
            self.showImage.save('origin.jpg')
        else:
            QMessageBox.critical(self, '错误', '摄像头未打开！')
            return None

    def duibi(self):
        name=self.comboBox_duibi.currentText()
        white="cv2.THRESH_BINARY_INV"
        black="cv2.THRESH_BINARY"
        if name=="亮色背景深色物体":
            FileChange("DUIBI",black)
        elif name=="深色背景亮色物体":
            FileChange("DUIBI",white)

    def caculater(self):
        #nob=FileChange("morden",None)
        angle=readAngle()
        sign=FileChange("morden",None)
        if sign=="2":#PL
            point=int(FileChange("point_PL",None))
            line=int(FileChange("line_PL",None))
            number=Calculate(point,line,angle,sign,None)
        elif sign=="1":#LL
            line1=int(FileChange("line_LL_F",None))
            line2=int(FileChange("line_LL_S",None))
            number=Calculate(line1,line2,angle,sign,self.data)
        elif sign=="0":#PP
            point1=int(FileChange("point_PP_F",None))
            point2=int(FileChange("point_PP_S",None))
            number=Calculate(point1,point2,angle,sign,None)
        self.label_pixel.setText(str(number))
        data=FileChange("data",None)
        en=float(number)/float(data)
        self.output.setText(str(en))

    def lastline(self,NOB):
        sign=FileChange("morden",None)
        angle=readAngle()
        path="NewPicture.jpg"
        show="OptionPicture.jpg"
        if sign=="2":
            point=int(FileChange("point_PL",None))
            line=int(FileChange("line_PL",None))
            line=line-1      #控件行为，前一个线
            if line<0:
                line=len(angle)-1
            elif line>len(angle)-1:
                line=0
            self.label_line_show.setText(str(line+1))
            FileChange("line_PL",line)
        elif sign=="1":
            if NOB=="F":
                point=int(FileChange("line_LL_S",None))
                line=int(FileChange("line_LL_F",None))
                line=line-1      #控件行为，前一个线
                if line<0:
                    line=len(angle)-1
                elif line>len(angle)-1:
                    line=0
                self.label_line_show.setText(str(line+1))
                FileChange("line_LL_F",line)
            elif NOB=="S":
                point=int(FileChange("line_LL_F",None))
                line=int(FileChange("line_LL_S",None))
                line=line-1      #控件行为，前一个线
                if line<0:
                    line=len(angle)-1
                elif line>len(angle)-1:
                    line=0
                self.label_point_show.setText(str(line+1))
                FileChange("line_LL_S",line)
        Option(angle,path,point,line,sign,NOB)
        self.label_Phote.setPixmap(QPixmap(show))
    def nextline(self,NOB):
        sign=FileChange("morden",None)
        angle=readAngle()
        path="NewPicture.jpg"
        show="OptionPicture.jpg"
        if sign=="2":
            point=int(FileChange("point_PL",None))
            line=int(FileChange("line_PL",None))
            line=line+1       #控件行为，后一个线
            if line>len(angle)-1:
                line=0
            print(line)
            self.label_line_show.setText(str(line+1))
            FileChange("line_PL",line)
        elif sign=="1":
            if NOB=="F":
                point=int(FileChange("line_LL_S",None))
                line=int(FileChange("line_LL_F",None))
                line=line+1       #控件行为，后一个线
                if line>len(angle)-1:
                    line=0
                print(line)
                self.label_line_show.setText(str(line+1))
                FileChange("line_LL_F",line)
            elif NOB=="S":
                point=int(FileChange("line_LL_F",None))
                line=int(FileChange("line_LL_S",None))
                line=line+1       #控件行为，后一个线
                if line>len(angle)-1:
                    line=0
                print(line)
                self.label_point_show.setText(str(line+1))
                FileChange("line_LL_S",line)
        Option(angle,path,point,line,sign,NOB)
        self.label_Phote.setPixmap(QPixmap(show))

    def lastpoint(self,NOB):
        sign=FileChange("morden",None)
        angle=readAngle()
        path="NewPicture.jpg"
        show="OptionPicture.jpg"
        if sign=="2":
            point=int(FileChange("point_PL",None))
            line=int(FileChange("line_PL",None))
            point=point-1   #控件行为，前一个点
            if point<0:
                point=len(angle)-1
            elif point>len(angle)-1:
                point=0
            self.label_point_show.setText(str(point+1))
            FileChange("point_PL",point)
        elif sign=="0":
            if NOB=="F":
                point=int(FileChange("point_PP_F",None))
                line=int(FileChange("point_PP_S",None))
                point=point-1   #控件行为，前一个点
                if point<0:
                    point=len(angle)-1
                elif point>len(angle)-1:
                    point=0
                self.label_point_show.setText(str(point+1))
                FileChange("point_PP_F",point)

            elif NOB=="S":
                point=int(FileChange("point_PP_F",None))
                line=int(FileChange("point_PP_S",None))
                line=line-1   #控件行为，前一个点
                if line<0:
                    line=len(angle)-1
                elif line>len(angle)-1:
                    line=0
                self.label_line_show.setText(str(line+1))
                print(line,len(angle))
                FileChange("point_PP_S",line)
        Option(angle,path,point,line,sign,NOB)
        self.label_Phote.setPixmap(QPixmap(show))
    def nextpoint(self,NOB):
        sign=FileChange("morden",None)
        angle=readAngle()
        path="NewPicture.jpg"
        show="OptionPicture.jpg"
        if sign=="2":
            point=int(FileChange("point_PL",None))
            line=int(FileChange("line_PL",None))
            point=point+1       #控件行为，前一个点
            if point>len(angle)-1:
                point=0
            self.label_point_show.setText(str(point+1))
            FileChange("point_PL",point)
        if sign=="0":
            if NOB=="F":
                point=int(FileChange("point_PP_F",None))
                line=int(FileChange("point_PP_S",None))
                point=point+1       #控件行为，前一个点
                if point>len(angle)-1:
                    point=0
                self.label_point_show.setText(str(point+1))
                FileChange("point_PP_F",point)
            elif NOB=="S":
                point=int(FileChange("point_PP_F",None))
                line=int(FileChange("point_PP_S",None))
                line=line+1       #控件行为，前一个点
                if line>len(angle)-1:
                    line=0
                self.label_line_show.setText(str(line+1))
                FileChange("point_PP_S",line)
        Option(angle,path,point,line,sign,NOB)
        self.label_Phote.setPixmap(QPixmap(show))

    def StartDREW(self):
        tim1=time.time()
        parameter={"blurry":5,
        "median":120,
        "fit":3
        }#参数字典：模糊参数  二值中值  拟合精度
        parameter["blurry"]=FileChange("Blurry",None)
        parameter["median"]=FileChange("Median",None)
        parameter["fit"]=FileChange("Fit",None)
        path="origin.jpg"
        FileChange("point_PL","0")
        FileChange("line_PL","0")
        angle,self.data=ContourD(path,parameter["fit"],parameter["blurry"],parameter["median"])
        cx=FileChange("AX",None)
        cy=FileChange("AY",None)
        bx=FileChange("BX",None)
        by=FileChange("BY",None)
        Draw(angle,path,cx,cy,bx,by)
        self.label_Phote.setPixmap(QPixmap("NewPicture.jpg"))
        self.label_Phote.setScaledContents(True)
        tim2=time.time()
        self.speed_show_label.setText(str(tim2-tim1))
    def cun(self):
        num=self.lineEdit_data.text()
        FileChange("data",str(num))
#   显示文本
    def chang(self,name,text):
        num = name.value()
        text.setText(str(num))
#   贮存文本
    def lock(self,name,kong):
        num = kong.value()
        FileChange(name,str(num))

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)#自适应窗口分辨率
    app=QApplication(sys.argv)
    ui1=Login()
    #window.showFullScreen()
    #ui1.showFullScreen()
    ui1.show()
    t1=time.time()
    path="origin.jpg"#原始图片
    New="NewPicture.jpg"
    ken=cv2.imread(path)
    cv2.imwrite(New,ken)
    sys.exit(app.exec())
    en=input()


