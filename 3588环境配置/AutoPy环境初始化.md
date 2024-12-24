刷写固件 ubuntu

# **安装向日葵->自启动** 
sudo apt-get update
sudo apt-get install gnome-startup-applications
gnome-session-properties

# **安装py虚拟环境**
sudo apt install python3.8-venv
python3 -m venv [/path/to/new/virtual/environment] //创建环境
source [venv]/bin/activate //进入环境
Deactivate //退出环境


# **Linux永久修改pip配置源(可选）：**

**1.在根目录下创建.pip文件夹**
	mkdir ~/.pip
**2.在创建好的.pip文件夹下创建pip源配置文件**
	touch ~/.pip/pip.conf
**3.使用vim打开pip.conf配置文件**
	vim ~/.pip/pip.conf
**4.添加下述内容**
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com
```
**5. 保存退出->完成**


sudo apt install build-essential
sudo apt install cmake gcc g++

# 安装PyQt5:
## 前言
在嵌入式ARM 64位平台上，pip方式安装PyQt5时因pip源无编译好的aarch64的whl包，需要下载PyQt5和sip的源码包进行自动编译，通常会编译失败。
虽然sudo apt-get install python3-pyqt5可以安装成功，但apt方式安装的PyQt5包默认安装在usr/bin/python3/dist-packages下，且在多Python版本环境下编译成系统默认python版本的PyQt5库文件，当在virtualenv创建的不同Python版本的虚拟环境中时无法调用，故在嵌入式平台上采用源码编译方式安装PyQt5。
## 一、PyQt5及SIP简介
PyQt5 是Riverbank 公司的产品，分为开源版本和商业版本，开源版本就包含全部的功能。
SIP 是一个将C/C++库转换为Python 绑定的工具，SIP本来是为了开发PyQt 而开发的，现在也可以用于将任何C/C++库转换为Python 绑定。PyQt5依赖于SIP包。

## 二、开发部署环境
| 开发环境     | 软件版本/配置                                          |
| -------- | ------------------------------------------------ |
| 开发板型号    | Firefly AIO-3399 ProC开发板 (3+16GB)                |
| 开发板操作系统  | Ubuntu 18.04LTS                                  |
| 开发板固件版本  | AIO-RK3399PROC-UBUNTU18.04-GPT-20200525-1016.img |
| Python版本 | Python 3.7.10                                    |
| PyQt5版本  | 5.15.2                                           |
| SIP版本    | 4.19.25                                          |
| 电脑远程软件   | Xshell 7 & Xftpd 7                               |
注：PyQt5版本需与SIP版本对应，直接从[pip](https://edu.csdn.net/cloud/sd_summit?utm_source=glcblog&spm=1001.2101.3001.7020)源下载的源码包编译时会出错，  
需从riverbank官网下载：SIP包[下载链接](https://www.riverbankcomputing.com/static/Downloads/sip/4.19.25/sip-4.19.25.tar.gz)，PyQt5包[下载链接](https://pypi.tuna.tsinghua.edu.cn/packages/28/6c/640e3f5c734c296a7193079a86842a789edb7988dca39eab44579088a1d1/PyQt5-5.15.2.tar.gz)
## 三、PyQt5及SIP安装步骤

### 1.安装qt5-default

源码编译PyQt5时需要系统的/usr/lib/qt5/bin/路径下有qmake，需先安装qt5-default包，此时默认会安装qmake。
```
user@admin:~$ sudo apt-get install qt5-default
...
下列【新】软件包将被安装：
  libdrm-dev libegl1-mesa-dev libgl1-mesa-dev libgles1 libgles2-mesa-dev
  libglu1-mesa-dev libglvnd-core-dev libglvnd-dev libopengl0 libqt5concurrent5
  libqt5opengl5-dev libqt5sql5 libqt5sql5-sqlite libqt5test5 libqt5xml5
  libwayland-bin libwayland-dev libx11-xcb-dev libxcb-dri2-0-dev
  libxcb-dri3-dev libxcb-glx0-dev libxcb-present-dev libxcb-randr0-dev
  libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxshmfence-dev
  libxxf86vm-dev mesa-common-dev qt5-default qt5-qmake qt5-qmake-bin
  qtbase5-dev qtbase5-dev-tools qtchooser x11proto-xf86vidmode-dev
  ...
  #验证qmake是否已安装
user@admin:~$ qmake --version
QMake version 3.1

```
### 2.配置好python和虚拟环境
(暂时不用，可以先按最开始那段下载python)
按照[《Firefly AIO-3399ProC开发板安装RKNN Toolkit 1.6.0开发环境》](https://blog.csdn.net/foreverey/article/details/114400098?spm=1001.2014.3001.5501)文章配置python和虚拟环境。  
### 3.源码编译安装SIP包

安装编译所需软件包
```
sudo apt-get install cmake gcc g++ 
pip3 install --upgrade pip 
pip3 install wheel setuptools
```
编译SIP包
```bash
(pyqt5) user@admin:~$ cd ./pyqt5/
(pyqt5) user@admin:~$ tar zxvf sip-4.19.25.tar.gz
pyqt5) user@admin:~/pyqt5$ cd ./sip-4.19.25
(pyqt5) user@admin:~/pyqt5/sip-4.19.25$ sudo python3 configure.py --sip-module PyQt5.sip
This is SIP 4.19.25 for Python 3.7.12 on linux.
The SIP code generator will be installed in /usr/bin.
The sip.h header file will be installed in /usr/include/python3.7m.
The PyQt5.sip module will be installed in /usr/lib/python3/dist-packages/PyQt5.
The sip.pyi stub file will be installed in
/usr/lib/python3/dist-packages/PyQt5.
The default directory to install .sip files in is /usr/share/sip.
Creating sipconfig.py...
Creating top level Makefile...
Creating sip code generator Makefile...
Creating sip module Makefile...
(pyqt5) user@admin:~/pyqt5/sip-4.19.25$ sudo make
...
(pyqt5) user@admin:~/pyqt5/sip-4.19.25$ sudo make install
make[1]: 进入目录“/home/user/pyqt5/sip-4.19.25/sipgen”
cp -f sip /usr/bin/sip
cp -f /home/user/pyqt5/sip-4.19.25/siplib/sip.h /usr/include/python3.7m/sip.h
make[1]: 离开目录“/home/user/pyqt5/sip-4.19.25/sipgen”
make[1]: 进入目录“/home/user/pyqt5/sip-4.19.25/siplib”
cp -f sip.so /usr/lib/python3/dist-packages/PyQt5/sip.so
strip /usr/lib/python3/dist-packages/PyQt5/sip.so
cp -f /home/user/pyqt5/sip-4.19.25/sip.pyi /usr/lib/python3/dist-packages/PyQt5/sip.pyi
make[1]: 离开目录“/home/user/pyqt5/sip-4.19.25/siplib”
cp -f sipconfig.py /usr/lib/python3/dist-packages/sipconfig.py
cp -f /home/user/pyqt5/sip-4.19.25/sipdistutils.py /usr/lib/python3/dist-packages/sipdistutils.py
/usr/bin/python3 /home/user/pyqt5/sip-4.19.25/mk_distinfo.py "" /usr/lib/python3/dist-packages/PyQt5_sip-4.19.25.dist-info installed.txt
```

有可能源码编译pyqt5 fatal error python.h no such file or directory, 执行下面的语句补充python基础包，sudo apt-get install python3-dev可以改成对应的版本，例如sudo apt-get install python3.10-dev
```
sudo apt-get update 
sudo apt-get install python3-dev
```
### 4.源码编译安装PyQt5包

源码编译PyQt5时时间较长，若无报错耐心等待即可。
```bash
(pyqt5) user@admin:~$ cd ./pyqt5/
(pyqt5) user@admin:~$ tar zxvf PyQt5-5.15.2.tar.gz
(pyqt5) user@admin:~/pyqt5$ cd ./PyQt5-5.15.2
(pyqt5) user@admin:~/pyqt5/PyQt5-5.15.2$ sudo python3 configure.py
Querying qmake about your Qt installation...
Determining the details of your Qt installation...
This is the GPL version of PyQt 5.15.2 (licensed under the GNU General Public
License) for Python 3.7.12 on linux.

Type 'L' to view the license.
Type 'yes' to accept the terms of the license.
Type 'no' to decline the terms of the license.

Do you accept the terms of the license? yes
...
(pyqt5) user@admin:~/pyqt5/PyQt5-5.15.2$ sudo make -j4
...
g++ -Wl,--version-script=pyrcc.exp -Wl,-O1 -shared -o libpyrcc.so sippyrccRCCResourceLibrary.o sippyrcccmodule.o rcc.o  -lQt5Xml -lQt5Core -lpthread  
cp -f libpyrcc.so pyrcc.so
make[1]: 离开目录“/home/user/pyqt5/PyQt5-5.15.2/pyrcc”
cd Qt/ && ( test -e Makefile || /usr/lib/qt5/bin/qmake -o Makefile /home/user/pyqt5/PyQt5-5.15.2/Qt/Qt.pro ) && make -f Makefile 
make[1]: 进入目录“/home/user/pyqt5/PyQt5-5.15.2/Qt”
gcc -c -pipe -O2 -fno-exceptions -Wall -W -D_REENTRANT -fPIC -DSIP_PROTECTED_IS_PUBLIC -Dprotected=public -DQT_NO_EXCEPTIONS -DQT_NO_DEBUG -DQT_PLUGIN -I. -I. -isystem /usr/include/python3.7m -I/usr/lib/x86_64-linux-gnu/qt5/mkspecs/linux-g++ -o sipQtcmodule.o sipQtcmodule.c
rm -f libQt.so
g++ -Wl,--version-script=Qt.exp -Wl,-O1 -shared -o libQt.so sipQtcmodule.o  -lpthread  
cp -f libQt.so Qt.so
make[1]: 离开目录“/home/user/pyqt5/PyQt5-5.15.2/Qt”
(pyqt5) user@admin:~/pyqt5/PyQt5-5.15.2$ sudo make install 
...
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtPrintSupport.pyi /usr/lib/python3/dist-packages/PyQt5/QtPrintSupport.pyi
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtSql.pyi /usr/lib/python3/dist-packages/PyQt5/QtSql.pyi
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtTest.pyi /usr/lib/python3/dist-packages/PyQt5/QtTest.pyi
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtWidgets.pyi /usr/lib/python3/dist-packages/PyQt5/QtWidgets.pyi
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtXml.pyi /usr/lib/python3/dist-packages/PyQt5/QtXml.pyi
/usr/lib/qt5/bin/qmake -install qinstall /home/user/pyqt5/PyQt5-5.15.2/QtDBus.pyi /usr/lib/python3/dist-packages/PyQt5/QtDBus.pyi
/usr/bin/python3 /home/user/pyqt5/PyQt5-5.15.2/mk_distinfo.py "" /usr/lib/python3/dist-packages/PyQt5-5.15.2.dist-info installed.txt
...
```
有可能遇到： fatal error GLES/gl32.h: No such file or directory。 错误通常是因为在编译包含 OpenGL ES 的项目时，缺少了相关的开发头文件和库。这些文件对于构建使用 OpenGL ES 的应用程序是必需的。
#### 1. 安装 OpenGL ES 开发库（目前使用这个可行）

大部分情况下，缺少头文件是因为你的系统中没有安装与 OpenGL ES 相关的开发库。

##### **Ubuntu/Debian 系统** 

你可以通过安装与 OpenGL ES 相关的开发库来解决此问题：
```bash
sudo apt-get update 
sudo apt-get install libgl1-mesa-dev libgles2-mesa-dev
```

- `libgl1-mesa-dev`：这是与 OpenGL 的 Mesa 开发包。
- `libgles2-mesa-dev`：这是与 OpenGL ES 相关的 Mesa 开发包，它包含了 `GLES` 头文件。

##### **Fedora/RedHat 系统**

对于 Fedora 或基于 RedHat 的系统，你可以使用以下命令：
`sudo dnf install mesa-libGLES-devel mesa-libGL-devel`
##### **openSUSE 系统**

对于 openSUSE 系统，安装如下开发包：
`sudo zypper install Mesa-libGLES-devel Mesa-libGL-devel`

#### 2. 检查 Include 路径

在进行编译时，如果头文件在系统中的位置无法被编译器自动找到，你可能需要手动指定这些文件的包含路径。假设头文件位于 `/usr/include` 或 `/usr/include/GLES`，你可以在编译命令中添加 `-I` 参数来指示编译器查找这些路径，例如：
`g++ your_code.cpp -o output -I/usr/include -lGLESv2`
#### 3. 检查 `gl32.h` 文件的可用性

在某些情况下，`gl32.h` 可能并不是系统中标准安装的文件。这可能与 OpenGL 版本有关，因为 OpenGL ES 3.2 可能不是默认包含的部分。如果你明确需要 OpenGL ES 3.2 的功能，你可以尝试以下几步：

- **验证 OpenGL 版本**：确认你是否安装了支持 OpenGL ES 3.2 的 Mesa 或图形驱动程序。
- **安装更新的 Mesa 库**：某些情况下，安装更新版本的 Mesa 可能包含所需的头文件。

#### 4. 使用其他图形 API 替代方案

如果你的项目不是强制使用 OpenGL ES 3.2，可能考虑使用更为普遍的 OpenGL 版本（如 3.0 或 2.0）。对于很多桌面平台，OpenGL 2.0 或 3.0 已经足够且更容易获得。

#### 5. 安装全套开发库

在某些情况下，缺少与图形相关的其他依赖项也可能导致类似的问题。你可以尝试安装整个 Mesa 和开发工具包：
`sudo apt-get install mesa-common-dev`
这将确保系统中安装所有与 Mesa 和 OpenGL 相关的开发头文件和库。

## 四、虚拟环境中调用PyQt5的配置

进入PyQt5源码编译安装的路径，打包后移动至虚拟环境的dist-packages下并解压。
```
(pyqt5) user@admi:~/pyqt5/PyQt5-5.15.2$ cd /usr/lib/python3/dist-packages/
(pyqt5) user@admin:/usr/lib/python3/dist-packages$ sudo tar zcvf pyqt5.tar.gz ./PyQt5/
./PyQt5/
./PyQt5/pyrcc.so
./PyQt5/QtDBus.pyi
./PyQt5/QtSql.so
./PyQt5/pyrcc_main.py
./PyQt5/QtSql.pyi
./PyQt5/QtNetwork.pyi
./PyQt5/QtNetwork.so
./PyQt5/sip.so
./PyQt5/QtGui.so
./PyQt5/_QOpenGLFunctions_4_1_Core.so
./PyQt5/QtDBus.so
./PyQt5/QtWidgets.pyi
./PyQt5/QtTest.so
./PyQt5/QtOpenGL.so
./PyQt5/QtOpenGL.pyi
./PyQt5/pylupdate.so
./PyQt5/_QOpenGLFunctions_2_1.so
./PyQt5/uic/
./PyQt5/uic/properties.py
./PyQt5/uic/exceptions.py
./PyQt5/uic/Loader/
./PyQt5/uic/Loader/loader.py
./PyQt5/uic/Loader/qobjectcreator.py
./PyQt5/uic/Loader/__init__.py
./PyQt5/uic/pyuic.py
./PyQt5/uic/Compiler/
./PyQt5/uic/Compiler/misc.py
./PyQt5/uic/Compiler/qtproxies.py
./PyQt5/uic/Compiler/qobjectcreator.py
./PyQt5/uic/Compiler/indenter.py
./PyQt5/uic/Compiler/proxy_metaclass.py
./PyQt5/uic/Compiler/__init__.py
./PyQt5/uic/Compiler/compiler.py
./PyQt5/uic/port_v2/
./PyQt5/uic/port_v2/proxy_base.py
./PyQt5/uic/port_v2/string_io.py
./PyQt5/uic/port_v2/as_string.py
./PyQt5/uic/port_v2/__init__.py
./PyQt5/uic/port_v2/ascii_upper.py
./PyQt5/uic/driver.py
./PyQt5/uic/icon_cache.py
./PyQt5/uic/__init__.py
./PyQt5/uic/widget-plugins/
./PyQt5/uic/widget-plugins/qtcharts.py
./PyQt5/uic/widget-plugins/qaxcontainer.py
./PyQt5/uic/widget-plugins/qtwebkit.py
./PyQt5/uic/widget-plugins/qtquickwidgets.py
./PyQt5/uic/widget-plugins/qtwebenginewidgets.py
./PyQt5/uic/widget-plugins/qtprintsupport.py
./PyQt5/uic/widget-plugins/qscintilla.py
./PyQt5/uic/port_v3/
./PyQt5/uic/port_v3/proxy_base.py
./PyQt5/uic/port_v3/string_io.py
./PyQt5/uic/port_v3/as_string.py
./PyQt5/uic/port_v3/__init__.py
./PyQt5/uic/port_v3/ascii_upper.py
./PyQt5/uic/uiparser.py
./PyQt5/uic/objcreator.py
./PyQt5/_QOpenGLFunctions_2_0.so
./PyQt5/QtPrintSupport.so
./PyQt5/QtXml.so
./PyQt5/__init__.py
./PyQt5/sip.pyi
./PyQt5/pylupdate_main.py
./PyQt5/QtGui.pyi
./PyQt5/QtXml.pyi
./PyQt5/QtWidgets.so
./PyQt5/QtCore.pyi
./PyQt5/QtPrintSupport.pyi
./PyQt5/QtTest.pyi
./PyQt5/Qt.so
./PyQt5/QtCore.so
(pyqt5) user@admin:/usr/lib/python3/dist-packages$ sudo cp ./pyqt5.tar.gz /home/user/pyqt5/lib/python3.7/site-packages/
(pyqt5) user@admin:~$ cd /home/user/pyqt5/lib/python3.7/site-packages/
(pyqt5) user@admin:~/pyqt5/lib/python3.7/site-packages$ tar zxvf pyqt5.tar.gz
...
```

至此完成PyQt5的安装

# 安装ff_pymedia
官方源码：[ffmedia_release](https://gitlab.com/firefly-linux/ffmedia_release)

安装音频相关模块依赖库

```
sudo apt install libasound2-dev libfdk-aac-dev
```
