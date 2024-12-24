I just installed a new Ubuntu 14.04 on my laptop, and downloaded the <u>JetPack3.0</u>. Followed instructions, and finished it successfully.

However when I try to <u>apt-get update</u> on the host computer I get the following error:

>W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/trusty-updates/main/binary-arm64/Packages  404  Not Found [IP: 91.189.88.162 80]  

There has been posts about this on this forum, but none of them has the solution. Please notice that this is not the Date problem with the release files. For some reason ubuntu mirrors does not have binary-arm64 packages.


# 解决方法：
 added [arch=amd64,i386] for each line in /etc/apt/sources.list starting with 'deb ’ . (for deb-src it would have no sense).

## 首先介绍ubuntu apt换成国内源

### 1. **备份源文件**
```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
```

### 2. **修改源文件**

```bash
sudo vim /etc/apt/sources.list
```

```text
deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
```

### 3.更新源文件
```bash
sudo apt update
```

# # 解决Ubuntu 找不到ARM64 的源的问题
从
```text
deb [http://archive.ubuntu.com/ubuntu](http://archive.ubuntu.com/ubuntu) xenial universe
```
改成
```text
deb [arch=amd64,i386] [http://archive.ubuntu.com/ubuntu](http://archive.ubuntu.com/ubuntu) xenial universe
```
