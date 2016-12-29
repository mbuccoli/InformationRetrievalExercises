import numpy as np

alphabet=['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India']
A=[]
def generate_exercize():
    A=np.zeros((len(alphabet),2))
    for a in range(len(alphabet)):
        A[a,:]=np.random.randn(2)*1.5+np.random.rand(1)*5+2.5

    A[A<0]=0.05
    A[A>10]=10
    
    return A
    
                 
def propose_exercize():
    global A
    A=generate_exercize()
    print('A query is proposed to a generic IR system and the following rates are obtained\n')
    print('Result\t\tRank1\t\tRank2')
    for i_a, a in enumerate(alphabet):
        print('%s\t\t%.2f\t\t%.2f'%(a,A[i_a,0],A[i_a,1]))
        
    print('Compute the top-3 results using the MedRank, the Fagin\'s and the Fagin\'s threshold algorithms')
    return A
    
    
    
def sort_A(A):
    S=[]
    Alph=np.array(alphabet)
    R=[]
    for i in range(A.shape[1]):
        idxs=np.flipud(np.argsort(A[:,i]))
        R.append(Alph[idxs])
        S.append(A[idxs,i])
    return np.array(S).T, np.array(R).T

def solve_medrank(S,R, debug=False):
    num={}
    for a in alphabet:
        num[a]=0
    k=0
    K=0
    Res=[]
    while len(Res)<3:
        if k!=0 and debug:
            print('List has not been completely filled, another iteration is needed\n')    
        if debug:
            print('Rank position: %d'%(k+1))
        for j in range(2):
            if debug:
                print('Analyzing %s'%R[k,j])
            num[R[k,j]]+=1

            if num[R[k,j]]==2:
                if debug:
                    print('%s was found in more than half of the ranks: adding in the list'%R[k,j])
                Res.append(R[k,j])

        k=k+1
    print('=== MedRank ===')
    for i in range(3):
        print(Res[i])
    
    return Res,k

def solve_Fagin(S,R,k, debug=False):

    alph=np.unique(R[:k,:])

    if debug:
        print('From Medrank, we know that we need to consider the first %d positions with sequential access'%k)

    A_alph=np.array([A[alphabet.index(a),:] for a in alph])
    mean_alph=np.mean(A_alph,axis=1)
    if debug:
        print('Using random access to retrieve the scores')
        print('The results are %s'%str(alph))
        print('With the following average score')
        for i in range(mean_alph.size):
            print('%s\t%.2f'%(alph[i],mean_alph[i]))

    idxs=np.flipud(np.argsort(mean_alph))
    print('=== Fagin\'s algorithm ===')
    print('Rank\tScore')
    for i in range(3):
        print('%s\t%.2f'%(alph[idxs[i]],mean_alph[idxs[i]]))
    
class Results(object):
    def __init__(self, max_items=3):
        self.max_items=3
        self.elements=[]
        self.scores=[]
        self.num=0
    def put(self, name, score):
        if self.num==0:
            self.elements.append(name)
            self.scores.append(score)
            self.num=1
            return
        new_el=[]
        new_sc=[]
        score_inserted=False
        for i in range(self.num):
            if self.scores[i]<score and not score_inserted:
                new_el.append(name)
                new_sc.append(score)
                score_inserted=True
                
            new_el.append(self.elements[i])
            new_sc.append(self.scores[i])
        if not score_inserted:
            new_el.append(name)
            new_sc.append(score)
                
        if len(new_el)>self.max_items:
            new_el=new_el[:self.max_items]
            new_sc=new_sc[:self.max_items]
        self.elements=new_el
        self.scores=new_sc
        self.num=len(self.elements)
        
    def min_score(self):
        return self.scores[-1]
    def print(self):
        print('\n'.join(['%d\t%s\t%.2f'%(k, self.elements[k],self.scores[k]) for k in range(self.num) ]))
def solve_Fagin_th(S,R, debug=False):
    th=15
    res=Results(3)
    seen={}
        
    for k in range(A.shape[0]):
        if debug:
            print('\n==> Rank position: %d'%(k+1))
        for j in range(A.shape[1]):
            j2=1-j
            a=R[k,j]
            if a in seen:
                continue
            k2=np.where(R[:,j2]==a)
            score=float(.5*(S[k,j]+S[k2,j2]))
            seen[a]=0            
            if debug:
                print('%s - score %.2f' %(a,score))
            if res.num<3 or score>min_score:
                if debug:
                    print('List not fill or score higher than the minimum')
                    print('Inserting %s'%a)
                res.put(a,score)
                min_score=res.min_score()
            if debug:
                print('Corrent top-K list')
                res.print()
                print()
        th=np.mean(S[k,:])
        if debug:
            print('Threshold: %.2f'%th)
        if res.num>=3 and min_score>=th:
            if debug:                
                print('Min score is higher than the threshold and the list was filled: finished\n')
            break        
    print('=== Fagin\'s threshold algorithm ===')
    print('Rank\tScore')
    for i in range(3):
        print('%s\t%.2f'%(res.elements[i],res.scores[i]))
    
        
def solve_exercize(A, debug=False):
    S,R=sort_A(A)
    print('First, let us sort the two rankings\n')
    print('Rank1\tScore1\t\tRank2\tScore2')
    for i_a in range(A.shape[0]):
        print('%s\t%.2f\t\t%s\t%.2f'%(R[i_a,0],S[i_a,0],R[i_a,1],S[i_a,1]))
    print('\n\n')
    Res,k=solve_medrank(S,R,debug)
    print('\n\n')
    solve_Fagin(S,R,k,debug)
    print('\n\n')
    solve_Fagin_th(S,R, debug)
    
    