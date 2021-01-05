import os
import sys

import argparse
import random
parser=argparse.ArgumentParser()
parser.add_argument('--allstar',type=str,default='all.star')
parser.add_argument('--goodstar',type=str,default='good.star')
parser.add_argument('--headstar',type=str,default='head.star')
parser.add_argument('--sets',type=int,default=20)
parser.add_argument('--outputdir',type=str,default='seed_out')
parser.add_argument('--shell',type=str,default='cls.sh')
parser.add_argument('--iter',type=str,default='30')
parser.add_argument('--heartbeat',type=str,default='3')
parser.add_argument('--skip_split',type=int,default=0)
parser.add_argument('--star_dir',type=str,default='seed_star')
parser.add_argument('--gpus',type=str,default='0,1,2,3,4,5,6,7')
parser.add_argument('--ref',type=str)
parser.add_argument('--inith',type=str,default='15')
args=parser.parse_args()

def cmd_exec(cmd):
  tmp_p=os.popen(cmd).read().split('\n')
  tmp_p.remove('')
  return tmp_p

def get_id(modelname): 
  tmp_split_run=modelname.split('run')[1]
  tmp_id=tmp_split_run.split('_')[0]
  job_id=tmp_id
  return job_id

def write_list(file_p,list_input):
  for eachline in list_input:
    file_p.write(eachline)


def open_star(filename):
  file_point=open(filename)
  file_return=file_point.read().split('\n')
  file_return.remove('')
  return file_return

def get_gpu():
  tmp=cmd_exec("nvidia-smi |grep MiB |grep -v / |awk '{print($2)}'")
  return tmp

def get_job():
  tmp=cmd_exec("ps -aux | grep relion_refine| grep mpirun| grep -v color |awk '{if (!x[$2]++) print $2}'")
  return tmp


def job2gpu(jobid):
  tmp=cmd_exec("ps -aux |grep mpi |grep "+jobid+"|awk -F '' '$0=$NF'")
  return tmp[0]


def get_status():
  current_run=[]
  for jobs in get_job():
    current_run.append(jobs)
  return current_run

def init_gpu():
  current_run=get_status()
  gpu_list={}
  for gpu in gpus:
    gpu_list[gpu]="-1"

  for eachjob in current_run:
    if job2gpu(eachjob) in gpus:
      gpu_list[job2gpu(eachjob)]=eachjob 
 
  return gpu_list



sets=args.sets


gpu_input=args.gpus
gpus=gpu_input.split(',')
max_jobs=len(gpus)

gpu_list=init_gpu()

os.system('mk.sh '+args.star_dir)

if (args.skip_split!=1):
# split star
#read all.star
  all_star_lines=open_star(args.allstar)
  random.shuffle(all_star_lines)
#print(all_star_lines)
  all_len=len(all_star_lines)
#======read good.star
  good_lines=open_star(args.goodstar)

  split_num=args.sets-1 
  patch=int(all_len/sets)
  #print(patch)
  for si in range(args.sets-1):
    tmp_str=args.star_dir+'/seed_input_'+str(si)+'.star'
    os.system('cat '+args.headstar+' >'+tmp_str)
    out_file=open(tmp_str,'a+')
    lines_patch=all_star_lines[si*patch+1:(si+1)*patch+1]
    lines_patch+=good_lines
    random.shuffle(lines_patch)
    for eachline in lines_patch:
      out_file.write(eachline)
      out_file.write('\n')
      out_file.flush()


    out_file.close()
# remainder
  si+=1
  tmp_str=args.star_dir+'/seed_input_'+str(si)+'.star'
  os.system('cat '+args.headstar +'>'+tmp_str)
  out_file=open(tmp_str,'a+')
  lines_patch=all_star_lines[si*patch+1:]
  lines_patch+=good_lines
  random.shuffle(lines_patch)
  for eachline in lines_patch:
    out_file.write(eachline)
    out_file.write('\n')
    out_file.flush()

job_queue=[]

for jobid in range(sets):
  cmd='sh '+args.shell+' '+args.outputdir+' '+str(jobid)+' '+args.star_dir+'/seed_input_'+str(jobid)+'.star '+args.iter+' '+args.ref+' '+args.inith+' '
  # gpu id not assigned
  job_queue.append(cmd)



if (args.skip_split==2):
  sys.exit(1)

def check_gpu(gpu_list):
  current_sys_gpu=get_gpu()
  free_gpu=[]
  for gpu in gpus:
    if gpu not in current_sys_gpu and gpu_list[gpu]=='-1':
      free_gpu.append(gpu)
  return free_gpu


def check_job(jobid):
  tmp=cmd_exec("ps -aux |grep relion_refine |grep -v ps |grep "+jobid)
  #print('check job state of '+jobid+'get return :',tmp)
  return tmp[0]

free_gpu=check_gpu(gpu_list)
#print(gpu_list) 

while(len(job_queue)>0):
  current_run=get_status()
  print(current_run)
  print(free_gpu)
  print(gpu_list)  

  if len(free_gpu)>0 and len(current_run)<max_jobs:
    # assign a gpu and run
    tmp_gpu_id=free_gpu[0]
    print(tmp_gpu_id)
    #free_gpu.remove(tmp_gpu_id)

    # do not remove gpuid until really assigned
    tmp_job_cmd=job_queue[0]
    tmp_job_id=tmp_job_cmd.split('seed_input_')[1].split('.')[0]
    
    check_finished_cmd='ls Class3D/'+args.outputdir+'/run'+str(tmp_job_id)+'_it0'+args.iter+'_model.star'
    try:
      print ('job '+get_id(cmd_exec(check_finished_cmd)[0])+' has finished and will be skipped.') 
      job_queue.remove(tmp_job_cmd)
      continue
    except:      
      cmd=tmp_job_cmd+str(tmp_gpu_id)
      print(cmd)
      os.system('sleep 1')
      job_queue.remove(tmp_job_cmd)
      os.popen(cmd)
      current_run=get_status()
      free_gpu.remove(tmp_gpu_id)

      #print(current_run)
      os.system('sleep 1')
      for eachjob in current_run:
        if job2gpu(eachjob) in gpus:
          gpu_list[job2gpu(eachjob)]=eachjob 
          
  else:
    # sleeping
    print('Sleeping ...')
    for gpu in gpus:
      try:
        check_job(gpu_list[gpu])
      except:
        gpu_list[gpu]='-1'
    # check old job (and free)
    current_run=get_status()
    os.system('sleep 1')
    for eachjob in current_run:
      if job2gpu(eachjob) in gpus:
        gpu_list[job2gpu(eachjob)]=eachjob 
    free_gpu=check_gpu(gpu_list)
    # set new gpu_list
    os.system('sleep '+args.heartbeat)
    
