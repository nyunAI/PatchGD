default = {
    "head" : 'original',
    "grad_accumulation" : True,
    "latent_size": 256,
    "sampling_batch_size":(1/8,30)
}
jobs = [default]

sampling = [1/4,1/8,1/16,1/32]
batch_size = [10,20,30,40]
modifications = {
    "head" : ['smaller','larger', 'smaller_feat', 'larger_feat'],
    "grad_accumulation" : [True,False],
    "latent_size": [128,512],
    "sampling_batch_size": [(i,j) for i,j in zip(sampling,batch_size)]
}

for key in modifications:
    for v in modifications[key]:
        k = default.copy()
        k[key] = v
        jobs.append(k)

for i in jobs:
    print(i)