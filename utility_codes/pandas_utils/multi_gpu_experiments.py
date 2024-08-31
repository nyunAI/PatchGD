import subprocess as sp
from multiprocessing.dummy import Pool
import sys, os

NR_GPUS = 8
NR_PROCESSES = 16

cnt = -1

def call_script(args):
    global cnt

    head = args['head']
    grad_accumlation = args['grad_accumulation']
    latent_size = args['latent_size']
    sampling_batch_size = args['sampling_batch_size']
    sampling = sampling_batch_size[0]
    batch_size = sampling_batch_size[1]
    
    crt_env = os.environ.copy()
    crt_env['OMP_NUM_THREADS'] = '1'
    crt_env['MKL_NUM_THREADS'] = '1'
    crt_env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    cnt += 1
    gpu = cnt % NR_GPUS
    crt_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    sp.call([sys.executable, './main.py',
                            '--gpu_id',str(gpu),
                            '--head',head,
                            '--grad_accumulation',str(grad_accumlation),
                            '--latent_size', str(latent_size), 
                            '--sampling',str(sampling),
                            "--batch_size",str(batch_size),
                            '--monitor_wandb',str(True)
                            ], env=crt_env)
    
if __name__ == '__main__':
    pool = Pool(NR_PROCESSES)
    default = {
        "head" : 'original',
        "grad_accumulation" : True,
        "latent_size": 256,
        "sampling_batch_size":(1/8,32)
    }

    jobs = [default]

    if True:
        sampling = [1/4,1/16,1/32]
        batch_size = [16,60,104]
        modifications = {
            "head" : ['smaller','larger', 'smaller_feat', 'larger_feat'],
            "grad_accumulation" : [False],
            "latent_size": [128,512],
            "sampling_batch_size": [(i,j) for i,j in zip(sampling,batch_size)]
        }

        for key in modifications:
            for v in modifications[key]:
                k = default.copy()
                k[key] = v
                jobs.append(k)
    
    args = jobs
    print(args,'\n')
    pool.map(call_script, args)
    pool.close()
    pool.join()