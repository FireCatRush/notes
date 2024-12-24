## 目录
- [磁盘信息查看](#磁盘信息查看)
- [磁盘分区管理](#磁盘分区管理)
- [挂载管理](#挂载管理)
- [自动挂载配置](#自动挂载配置)
- [常见问题排查](#常见问题排查)
- [实用命令速查](#实用命令速查)

## 磁盘信息查看

### 基础命令
```bash
# 查看磁盘设备列表
lsblk

# 查看磁盘大小和型号
lsblk -d -o NAME,TYPE,SIZE,MODEL

# 查看分区的挂载点及大小
lsblk -o NAME,SIZE,MOUNTPOINT

# 查看分区详细信息
sudo fdisk -l /dev/nvme0n1

# 检查磁盘是否是MBR或GPT
sudo parted /dev/nvme0n1 print

# 查看磁盘UUID和标签
sudo blkid
```

### 磁盘使用情况查看
```bash
# 查看文件系统使用情况
df -h

# 查看目录占用空间
du -sh /path/to/directory

# 查看当前目录下各子目录大小
du -h --max-depth=1
```

## 磁盘分区管理

### 分区类型说明
- **主分区(Primary)**: 最多可创建4个
- **扩展分区(Extended)**: 作为逻辑分区的容器
- **逻辑分区(Logical)**: 在扩展分区内创建，数量不限

### 分区操作步骤
1. 启动分区工具：
```bash
sudo fdisk /dev/nvme0n1
```

2. 常用fdisk命令：
- `p`: 打印分区表
- `n`: 新建分区
- `d`: 删除分区
- `t`: 更改分区类型
- `w`: 写入更改并退出
- `q`: 不保存退出

### 分区格式化
```bash
# 格式化为ext4文件系统
sudo mkfs.ext4 /dev/nvme0n1p3

# 格式化为FAT32文件系统
sudo mkfs.vfat -F 32 /dev/nvme0n1p1

# 格式化为NTFS文件系统
sudo mkfs.ntfs /dev/nvme0n1p4
```

## 挂载管理

### 基本挂载操作
```bash
# 创建挂载点
sudo mkdir /mnt/external_ssd

# 手动挂载
sudo mount /dev/nvme0n1p3 /mnt/external_ssd

# 卸载
sudo umount /mnt/external_ssd

# 强制卸载（当设备忙时）
sudo umount -l /mnt/external_ssd
```

### 挂载选项说明
- `rw`: 读写模式（默认）
- `ro`: 只读模式
- `noatime`: 不更新访问时间
- `defaults`: 默认选项（rw,suid,dev,exec,auto,nouser,async）
- `user`: 允许普通用户挂载
- `noexec`: 不允许执行二进制文件
- `sync`: 同步写入

## 自动挂载配置

### /etc/fstab配置
1. 获取分区UUID：
```bash
sudo blkid
```

2. 编辑fstab文件：
```bash
sudo nano /etc/fstab
```

3. 添加挂载配置：
```bash
# <file system>        <mount point>         <type>  <options>       <dump>  <pass>
UUID=xxxx-xxxx        /mnt/external_ssd     ext4    defaults        0       2
```

### 字段说明
- **file system**: 设备UUID或设备路径
- **mount point**: 挂载点
- **type**: 文件系统类型（ext4/ntfs/vfat等）
- **options**: 挂载选项
- **dump**: 备份选项（0表示不备份）
- **pass**: 开机检查顺序（0不检查，1根目录，2其他）

### 测试配置
```bash
# 测试fstab配置
sudo mount -a

# 验证挂载情况
df -h
```

## 常见问题排查

### 设备忙无法卸载
1. 查找占用进程：
```bash
sudo lsof /mnt/external_ssd
```

2. 结束占用进程：
```bash
sudo kill <PID>
```

### 磁盘空间不一致
1. 检查文件系统：
```bash
sudo fsck /dev/nvme0n1p3
```

2. 扩展文件系统：
```bash
sudo resize2fs /dev/nvme0n1p3
```

### 挂载点权限问题
```bash
# 修改挂载点权限
sudo chmod 755 /mnt/external_ssd

# 修改挂载点所有者
sudo chown user:group /mnt/external_ssd
```

## 实用命令速查

### 磁盘监控
```bash
# 实时监控IO状态
iostat -x 1

# 查看磁盘读写状态
iotop

# 查看SMART信息
sudo smartctl -a /dev/nvme0n1
```

### 性能优化
```bash
# 调整调度器
echo noop > /sys/block/nvme0n1/queue/scheduler

# 查看当前IO调度器
cat /sys/block/nvme0n1/queue/scheduler
```

### 备份与恢复
```bash
# 备份分区
sudo dd if=/dev/nvme0n1p1 of=/path/to/backup.img

# 恢复分区
sudo dd if=/path/to/backup.img of=/dev/nvme0n1p1
```

---
*注意：以上命令中的设备名称（如nvme0n1）请根据实际情况替换。在执行危险操作前请务必备份重要数据。*