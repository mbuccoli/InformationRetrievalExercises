import numpy as np
import matplotlib.pyplot as plt

def random_evaluations(nq=0, nd=0):
    if nq==0:
        nq=np.random.randint(2,4)
    Q=[]
    R=[]
    if nd==0:
        nd=np.random.randint(3,7)
    
    def generate(nd):
        keep=True
        while keep:
            Q_=1+np.random.permutation(nd)
            val=np.random.rand(1)*.2+.4
            R_=np.sign(np.random.rand(nd)-val)
            keep= np.abs(np.sum(R_))==nd
        return Q_, R_
    
    for q in range(nq):
        Q_, R_=generate(nd)
        R.append(R_) 
        Q.append(Q_)
    return Q, R

def propose_exercize(nq, nd):
    Q,R=random_evaluations(nq,nd)
    nq=len(Q)
    nd=len(Q[0])
    print('Your IR system received %d queries from users'%nq)
    print('Here are the top %d results your system retrieved for each query, '%nd+\
          'and the correct evaluation for each of them')
    header=['q%d\tRel\t\t'%(q+1) for q in range(nq)]
    print('d\t'+''.join(header))
    for d in range(nd):
        row=['%d\t%d\t\t'%(Q[q][d]+1,1 if R[q][d]==1 else 0) for q in range(nq)]
        print('#%d\t'%(d+1)+''.join(row))
    return Q, R

def solve(ex,Q,R):
    nq=len(Q)
    nd=len(Q[0])
    R_=np.array(R)
    R_=.5*(R_+1)
    Prec_tot=[]
    Rec_tot=[]    
    def compute_PR(print_screen=True):
        Prec_tot=[]
        Rec_tot=[]        
        if print_screen:
            print('Precision and Recall at k for k=1,...,%d'%nd)      
        for q in range(nq):
            q1=q+1
            r=R_[q,:]
            if print_screen:
                print('\tQuery %d'%q1)
            Prec_q=[]
            Rec_q=[]
            for k in range(nd):
                k1=k+1
                Prec=np.sum(r[:k1])/k1
                Rec=np.sum(r[:k1])/np.sum(r)                
                if print_screen:                    
                    print('\t\tP(%d)=%d/%d=%.2f,\tR(%d)=%d/%d=%.2f'\
                     %(k1, np.sum(r[:k1]), k1, Prec, k1, np.sum(r[:k1]),np.sum(r),Rec))
                Prec_q.append(Prec)
                Rec_q.append(Rec)
            Prec_tot.append(Prec_q)
            Rec_tot.append(Rec_q)
        Prec_tot=np.array(Prec_tot)
        Rec_tot=np.array(Rec_tot)
        return Prec_tot, Rec_tot
    def compute_TPFP(TP_rate=None):
        TP_tot=[]        
        FP_tot=[]        
        print('TP_rate and FP_rate at k for k=1,...,%d'%nd)      
        for q in range(nq):
            q1=q+1
            r=R_[q,:]
            nr=1-r
            print('\tQuery %d'%q1)
            TP_q=[]
            FP_q=[]
            for k in range(nd):
                k1=k+1
                TP=np.sum(r[:k1])/np.sum(r)                
                FP=np.sum(nr[:k1])/np.sum(nr)
                
                print('\t\tTP_rate(%d)=R(%d)=%d/%d=%.2f\t FP_rate(%d)=%d/%d=%.2f\t'\
                     %(k1, k1, np.sum(r[:k1]),np.sum(r),TP, k1,np.sum(nr[:k1]),np.sum(nr),FP))
                TP_q.append(TP)
                FP_q.append(FP)
            TP_tot.append(TP_q)
            FP_tot.append(FP_q)
        TP_tot=np.array(TP_tot)
        FP_tot=np.array(FP_tot)
        return TP_tot, FP_tot        
    
    if ex=='prec_rec' or ex=='all':        
        Prec_tot, Rec_tot=compute_PR()
        print('\n Draw the Precision-Recall curve for each query')  
        for q in range(nq):
            q1=q+1
            print('\tQuery %d'%q1)            
            plt.figure()
            Rec_q=Rec_tot[q,:]
            Prec_q=Prec_tot[q,:]
            plt.scatter(np.array(Rec_q), np.array(Prec_q))
            plt.plot(np.array(Rec_q), np.array(Prec_q),label='Precision-Recall curve')            
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('Recall'); plt.ylabel('Precision')
            R_int=np.hstack([0,Rec_q,1])
            P_int=np.zeros(R_int.size)
            for i_r in range(R_int.size-1):
                r=R_int[i_r]
                if i_r!=0 and R_int[i_r+1]==r:
                    P_int[i_r]=np.max(Prec_q[i_r-1:])    
                else:
                    P_int[i_r]=np.max(Prec_q[i_r:])            
            plt.plot(R_int,P_int,color='r',label='Interpolated PR curve')
            plt.legend(loc='lower left')
            plt.show()
    if ex=='r-prec' or ex=='all':        
        if Prec_tot==[]:
            Prec_tot, Rec_tot=compute_PR()
        print('\n Determine R-precision for each query') 
        for q in range(nq):            
            Rec_q=Rec_tot[q,:]
            Prec_q=Prec_tot[q,:]
            r=int(np.sum(R_[q]))
            q1=q+1
            print('\tQuery %d'%q1)
            print('\t\tNumber of relevant documents: %d --> P(%d)=%.2f'%(r,r,Prec_q[r-1]))
    if ex=='map' or ex=='all':        
        if Prec_tot==[]:
            Prec_tot, Rec_tot=compute_PR()
        print('\n Calculate the Mean Average Precision')
        APs=[]
        for q in range(nq):            
            Prec_q=Prec_tot[q,:]            
            r=int(np.sum(R_[q]))
            q1=q+1
            print('\tQuery %d'%q1)
            str_formula='1/%d '%r
            rs=np.where(R_[q]==1)[0]+1
            str_formula+='{' + ' + '.join(['P(%d)'%rs_ for rs_ in rs]) + '}'
            AP=np.mean(Prec_q[np.where(R_[q]==1)])            
            print('\t\tAP=%s=%.2f'%(str_formula, AP))
            APs.append(AP)
        
        APstring='1/%d {'%nq
        APstring+= ' + '.join(['AP_%d'%(q+1) for q in range(nq)]) 
        APstring+='}=1/%d {'%nq
        APstring+= ' + '.join(['%.2f'%(AP) for AP in APs]) 
        APstring+='}'        
        print('\tMAP=%s=%.2f'%(APstring, np.mean(np.array(APs))))
    if ex=='roc' or ex=='all' or ex=='auc':
        TP_tot, FP_tot=compute_TPFP()
        
        print('\n Draw the ROC curve for each query')  
        for q in range(nq):
            q1=q+1
            print('\tQuery %d'%q1)            
            plt.figure()
            TP_q=TP_tot[q,:]
            FP_q=FP_tot[q,:]
            plt.scatter(np.array(FP_q), np.array(TP_q))
            TP_q_=np.hstack([0,TP_q,1])
            FP_q_=np.hstack([0,FP_q,1])
            plt.plot(np.array(FP_q_), np.array(TP_q_),label='ROC curve')            
            plt.xlim([-0.05,1.05]); plt.ylim([-0.05,1.05])
            plt.xlabel('FP rate'); plt.ylabel('TP rate')
            plt.show()
            if ex=='auc' or ex=='all':
                AUC=[]
                for i_x in range(TP_q_.size-1):
                    delta_x=FP_q_[i_x+1]-FP_q_[i_x]
                    base=TP_q_[i_x+1]+TP_q_[i_x]
                    AUC.append(base*delta_x/2)
                AUC=np.array(AUC)
                AUC=AUC[AUC>0]
                string_AUC=' + '.join(['%.2f'%auc for auc in AUC])
                if string_AUC!='':
                    string_AUC+=' = '
                
                print('\tAUC = %s %.2f\n\n'%(string_AUC, np.sum(AUC)))            
    if ex=='clear':
        return
            