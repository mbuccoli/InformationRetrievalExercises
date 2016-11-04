import numpy as np
import matplotlib as plt

def check(d,D):
    for h in D:
        if np.linalg.norm(d-h)==0:
            return False
    return True        

def random_exercize(N=0,M=0):
    M=6
    N=4
    Q_ = [np.array([0,0,0,1,1,1])]
    D_ = [np.array([1,0,1,0,0,1]),
          np.array([1,0,1,0,0,0]),
          np.array([0,1,1,0,1,0]),
          np.array([1,0,0,1,1,0])]
    idxs=np.random.permutation(M)
    d_idxs=np.random.permutation(N)
    Q=[Q_[0][idxs]]
    D=[D_[d][idxs] for d in d_idxs]
    '''
    
    if N==0:
        N=np.random.randint(4,5)
    if M==0:
        M=np.random.randint(5,7)
    D=[]
    Q=[]
    i=0
    while i<N+1:
        d=np.zeros(M)
        h=np.random.permutation(M)
        num_1=np.random.randint(3,4)
        d[h[:num_1]]=1
        if i==0:
            Q.append(d)
        elif check(d,D):
            D.append(d)
        else:
            continue
        i=i+1
    '''
    return D,Q
def propose_exercize(N=0,M=0):
    D,Q=random_exercize(N,M)
    print('Given')
    print('\tQuery incidence vector')
    print('\t\tq=%s'%str(Q[0]))
    print('Documents incidence vectors')
    for i, d in enumerate(D):
        print('\t\td%d=%s'%(i+1,str(d)))
    print('\tInitialize')
    print('\t\tp_i=0.5')
    print('Consider as relevant the top-%d documents'%2)
    return D,Q

def reduce_terms(D,Q):
    print('Reduce the documents incidence vectors considering only the terms that appear in the query')
    idxs=np.where(Q[0]==1)[0]
    Q_new=[Q[0][idxs]]
    D_new=[d[idxs] for d in D]
    for i, d in enumerate(D_new):
        print('\t d%d=%s'%(i+1, str(d)))
    return D_new, Q_new

def initialize(D,Q):
    R={}
    print('\nInitialize u_i as n_i/N, for each term')
    N=len(D)
    T=Q[0].size
    n_i=np.zeros(T)
    for i in range(T):
        n=[d[i] for d in D]
        n_i[i]=np.sum(np.array(n))
    u_i=n_i/N
    print('\t%s'%'\t'.join(['t%d'%(i+1) for i in range(T)]))
    print('n_i\t%s'%'\t'.join(['%d'%n for n in n_i]))
    print('u_i\t%s'%'\t'.join(['%.2f'%u for u in u_i]))
    R['n']=n_i
    R['u']=u_i
    R['p']=.5*np.ones(n_i.shape)
    R['T']=T
    R['N']=N
    return R
    
def determine_SC(D,Q,R):
    SC=np.zeros(R['N'])
    for i_d, d in enumerate(D):
        idxs=np.where(d==1)[0]
        row='SC(d%d, q) = '%(i_d+1)        
        if len(idxs)==0:
            row+=' = -inf'        
            SC[i_d]=-np.inf
            print(row)
            continue
        logs=[]
        for i in idxs:
            x=i+1
            logs.append('log2(p%d/(1-p%d)) + log2((1-u%d)/u%d)'%(x,x,x,x))
        row+=' + '.join(logs)+' = '
        if np.any(R['p'][idxs]==1) or np.any(R['u'][idxs]==0):
            SC[i_d]=np.inf
            row+='+inf'
        elif np.any(R['p'][idxs]==0) or np.any(R['u'][idxs]==1):
            SC[i_d]=-np.inf        
            row+='-inf'
        else:
            SC[i_d]=np.sum(np.log2(R['p'][idxs]))-np.sum(np.log2(1-R['p'][idxs]))+\
                np.sum(np.log2(1-R['u'][idxs]))-np.sum(np.log2(R['u'][idxs]))
            row+='%.2f'%SC[i_d]
        print(row)
    R['SC']=SC
    return R
def rankSC(R):
    SC=R['SC']
    SC_rem=SC.copy()
    ranking=[]
    k=0
    while len(ranking)<R['N'] and k<R['N']:
        M=np.max(SC_rem)
        m=np.where(SC==M)[0]
        if m.size==1:        
            ranking.append(m[0])
        else:            
            n=m.size
            ranking.extend(list(m))
            
        SC_rem=SC_rem[SC_rem!=M]
        k+=1
    #ranking=np.flipud(np.argsort(R['SC']))
    
    print('\nRanking')
    print(' > '.join(['d%d'%(k+1) for k in ranking]))
    R['rank']=ranking
    return R

def update(D,R):
    S=2
    rel=R['rank'][:S]
    
    print('\nRelevant documents: {d%d, d%d}'%(rel[0]+1,rel[1]+1))
    s_i=np.zeros(R['n'].shape)
    row=''
    for i in range(R['T']):
        s_i[i]=np.sum(np.array([D[k][i] for k in rel]))
        row+='s_%d = %d\n'%(i+1,s_i[i])
    row+='$'
    print(row.replace('\n$',''))
    R['s']=s_i
    R['p']=s_i/S
    row=['p_%d = s_%d/S=%d/2=%.2f'%(i+1, i+1, R['s'][i], R['p'][i])  for i in range(R['T'])]
    print('\n'.join(row))
    u=(R['n']-R['s'])/(R['N']-S)
    R['u']=u
    row=['u_%d = (n_%d-s_%d)/(N-S)=(%d-%d)/(%d-2)=%.2f'%(i+1,i+1, i+1, R['n'][i], R['s'][i],R['N'], u[i])  for i in range(R['T'])]
    print('\n'.join(row))
    return R
    
def iterate(D,Q,R):
    R=determine_SC(D,Q,R)
    R= rankSC(R)
    return R
    
def check_convergence(R,old_R):
    S=2
    cur_rank=R['rank'][:S]
    old_rank=old_R[-1]['rank'][:S]
    for i in range(S):
        if cur_rank[i]!=old_rank[i]:
            return False
    return True


def solve_exercize(D,Q,clean):
    if clean:
        return
    D,Q=reduce_terms(D,Q)
    iteration=1
    
    R=initialize(D,Q)
    convergence=False
    old_R=[]
    while iteration<3 and not convergence:        
        print('\n\n===== Iteration #%d ====='%iteration)
        R=iterate(D,Q,R)
        
        if len(old_R)>0 and check_convergence(R,old_R):
            convergence=True
        else:            
            R=update(D,R)
            old_R.append(R)
        iteration+=1
    print('\nConvergence reached')