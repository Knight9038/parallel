import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import matplotlib.pyplot as plt

# 环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# 参数
NUM_NODES = 4  # 节点数
NUM_EPOCHS = 10  # 测试轮数
TENSOR_SIZE = 1000000  # 张量大小

def init_process(rank, size, fn, backend='nccl'):
    """ 初始化分布式环境 """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def parameter_server(rank, size):
    """ Parameter Server 模拟 """
    tensor = torch.ones(TENSOR_SIZE).cuda(rank)
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        if rank == 0:
            # 聚合梯度
            for i in range(1, size):
                recv_tensor = torch.zeros(TENSOR_SIZE).cuda(rank)
                dist.recv(recv_tensor, src=i)
                tensor += recv_tensor
            # 广播更新的参数
            for i in range(1, size):
                dist.send(tensor, dst=i)
        else:
            dist.send(tensor, dst=0)
            dist.recv(tensor, src=0)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch {epoch}, Rank {rank}, Time {elapsed_time:.4f}s')

def ring_allreduce(rank, size):
    """ Ring-AllReduce 模拟 """
    tensor = torch.ones(TENSOR_SIZE).cuda(rank)
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Epoch {epoch}, Rank {rank}, Time {elapsed_time:.4f}s')

def run_experiment(fn):
    """ 运行实验并收集时间数据 """
    size = NUM_NODES
    times = torch.zeros(NUM_EPOCHS, device='cuda:0')
    mp.spawn(init_process, args=(size, fn), nprocs=size, join=True)
    dist.barrier()
    return times.cpu().numpy()

def plot_results(ps_times, ring_times):
    """ 绘制结果图表 """
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, ps_times, label='Parameter Server')
    plt.plot(epochs, ring_times, label='Ring-AllReduce')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Parameter Server vs Ring-AllReduce Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == '__main__':
    ps_times = run_experiment(parameter_server)
    ring_times = run_experiment(ring_allreduce)
    plot_results(ps_times, ring_times)