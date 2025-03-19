import os
import json
import boto3
import paramiko
import time
from pathlib import Path

def setup_aws_instance(config_path):
    """设置AWS实例"""
    with open(config_path) as f:
        config = json.load(f)
    
    aws_config = config['aws']
    
    # 创建EC2客户端
    ec2 = boto3.client('ec2', region_name=aws_config['region'])
    
    # 启动实例
    response = ec2.run_instances(
        ImageId=aws_config['ami_id'],
        InstanceType=aws_config['instance_type'],
        KeyName=aws_config['key_name'],
        SecurityGroups=[aws_config['security_group']],
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': aws_config['volume_size'],
                    'VolumeType': 'gp3'
                }
            }
        ],
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': k, 'Value': v} for k, v in aws_config['tags'].items()
                ]
            }
        ]
    )
    
    instance_id = response['Instances'][0]['InstanceId']
    print(f"启动实例: {instance_id}")
    
    # 等待实例运行
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    
    # 获取实例公网IP
    response = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    
    return instance_id, public_ip

def setup_training_environment(public_ip, key_path):
    """设置训练环境"""
    # 创建SSH客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(public_ip, username='ubuntu', key_filename=key_path)
    
    # 创建SFTP客户端
    sftp = ssh.open_sftp()
    
    # 创建必要的目录
    commands = [
        'mkdir -p ~/kits23_data',
        'mkdir -p ~/kits23_output',
        'mkdir -p ~/kits23_code'
    ]
    
    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
    
    # 上传代码
    local_code_dir = Path(__file__).parent.parent
    remote_code_dir = '/home/ubuntu/kits23_code'
    
    for root, dirs, files in os.walk(local_code_dir):
        for dir_name in dirs:
            remote_dir = os.path.join(remote_code_dir, os.path.relpath(os.path.join(root, dir_name), local_code_dir))
            stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
        
        for file_name in files:
            local_file = os.path.join(root, file_name)
            remote_file = os.path.join(remote_code_dir, os.path.relpath(local_file, local_code_dir))
            sftp.put(local_file, remote_file)
    
    # 安装依赖
    commands = [
        'sudo apt-get update',
        'sudo apt-get install -y python3-pip',
        'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
        'pip3 install nibabel numpy pandas scikit-learn tensorboard',
        'pip3 install -e ~/kits23_code'
    ]
    
    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
    
    return ssh

def start_training(ssh, config_path):
    """启动训练"""
    # 上传配置文件
    sftp = ssh.open_sftp()
    sftp.put(config_path, '/home/ubuntu/kits23_code/config/cloud_config.json')
    
    # 启动训练
    cmd = 'cd ~/kits23_code && python3 main_train.py --use_cloud'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    # 实时输出训练日志
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line.strip())
    
    # 检查错误
    errors = stderr.read().decode()
    if errors:
        print("训练错误:")
        print(errors)

def main():
    # 配置文件路径
    config_path = Path(__file__).parent.parent / 'config' / 'cloud_config.json'
    key_path = os.path.expanduser('~/.ssh/kits23-key.pem')
    
    # 设置AWS实例
    instance_id, public_ip = setup_aws_instance(config_path)
    print(f"实例IP: {public_ip}")
    
    try:
        # 设置训练环境
        ssh = setup_training_environment(public_ip, key_path)
        
        # 启动训练
        start_training(ssh, config_path)
        
    finally:
        # 关闭SSH连接
        ssh.close()
        
        # 终止实例
        ec2 = boto3.client('ec2', region_name='us-east-1')
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"终止实例: {instance_id}")

if __name__ == "__main__":
    main() 