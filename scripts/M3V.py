import numpy as np
from numpy.ctypeslib import ndpointer
import os
import scipy.io as sio
from time import time
import ctypes

from scripts.utils.config_utils import config

lib = ctypes.cdll.LoadLibrary(os.path.join(config.scripts_root, "M3VLib.dll"))
incre_m3v_c = lib.train
incre_m3v_c.restype = ctypes.c_void_p
incre_m3v_c.argtypes = [ctypes.c_void_p,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

init = lib.init
init.restype = ctypes.c_void_p
init.argtypes =[ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def decom_similarity_matrix(compressed_similarity_matrix, dim):
    uncom_similarity_matrix = np.ones((dim, dim))
    count = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            uncom_similarity_matrix[i, j] = compressed_similarity_matrix[count]
            uncom_similarity_matrix[j, i] = compressed_similarity_matrix[count]
            count = count + 1
    return uncom_similarity_matrix


class CrowdsourcingModel(object):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10, v=1, alpha=1, TOL=1e-2, seed=None):
        self.l = l
        self.c = c
        self.n = n
        self.maxIter = maxIter
        self.burnIn = 0
        self.v = v
        self.alpha = alpha
        self.eps = 2.e-6
        self.verbose = 1
        self.trained = 0
        self.TOL = TOL
        self.seed = seed

    def from_numpy_data(self, L):
        self.L = L
        self.L = self.L.astype(np.int32)
        instance_num = self.L.shape[0]
        # self.true_labels = np.zeros(L.shape[0])
        self.true_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7] * (instance_num // 2 + 1))[:instance_num]
        self.backend_L = self.L.copy()
        self.true_labels = np.array(self.true_labels).reshape(-1).astype(np.int32) + 1
        self.origin_labels_num = (self.L > 0).sum()
        self.process_data()

    def loadData(self, filename):
        mat = sio.loadmat(filename)
        self.L = mat['L']
        try:
            self.L = self.L.toarray()
        except:
            self.L = self.L
        self.L = self.L.astype(np.int)
        self.true_labels = mat['true_labels'].reshape(-1).astype(np.int)
        self.process_data()


    def process_data(self):
        # get info from L for further convenience
        self.Ntask, self.Nwork = self.L.shape
        self.Ndom = len(set([i for i in self.true_labels.reshape(-1)]))
        self.LabelDomain = np.unique(self.L[self.L != 0])
        self.Ndom = len(self.LabelDomain)
        self.NeibTask = []
        for i in range(self.Ntask):
            tmp = [nt for nt in range(self.Nwork) if self.L[i, nt] > 0]
            self.NeibTask.append(tmp)
        self.NeibWork = []
        for j in range(self.Nwork):
            tmp = [nw for nw in range(self.Ntask) if self.L[nw, j] > 0]
            self.NeibWork.append(tmp)
        self.LabelTask = []
        for i in range(self.Ntask):
            tmp = [self.L[i, nt] for nt in self.NeibTask[i]]
            self.LabelTask.append(tmp)
        self.LabelWork = []
        for j in range(self.Nwork):
            tmp = [self.L[nw, j] for nw in self.NeibWork[j]]
            self.LabelWork.append(tmp)

    def cal_error_using_soft_label(self, mu, true_labels):
        '''
            to avoid ties, we take uniform probability over all calsses that
        maxmumize mu(classes, workers)
            1. average in case of ties
            2. Ignore when ture true_labels are NaN (missing)
        :param mu:
        :param true_labels:
        :return:
        '''
        index = (true_labels > 0)
        mu = mu[index, :]
        true_labels = true_labels[index]
        soft_label = mu / mu.sum(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        mu = (mu.max(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom) == mu)
        mu = mu.astype(np.float)
        self.posterior_labels = mu.argmax(axis=1) + 1
        mu = mu / mu.sum(axis=1).reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        tmp1 = np.array(range(1, 1 + self.Ndom)).reshape(1, -1).repeat(axis=0, repeats=true_labels.shape[0])
        tmpTrue = true_labels.reshape(-1, 1).repeat(axis=1, repeats=self.Ndom)
        error_rate = ((tmpTrue != tmp1) * mu).sum(axis=1).mean()
        soft_error_rate = ((tmpTrue != tmp1) * soft_label).sum(axis=1).mean()
        return error_rate, soft_error_rate, -1, -1

    def getsi(self, eta, i, k):
        mm = float('-inf')
        si = 0
        for kk in range(self.Ndom):
            # if kk != k
            if abs(kk - k) > self.eps:
                dx = self.X[i, kk, :]
                tmp = np.dot(eta, dx.T)
                if tmp > mm:
                    si = kk
                    mm = tmp
        return si + 1

class M3VModel(CrowdsourcingModel):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10,
                 v=1, alpha=1, TOL=1e-2, alpha2=35.649, beta2=6,
                 seed=None):
        super(M3VModel, self).__init__(l=l, c=c, n=n, maxIter=maxIter,
                                              burnIn=burnIn, v=v, alpha=alpha, TOL=TOL,
                                              seed=seed)
        self.alpha2 = alpha2
        self.beta2 = beta2

    def initial(self):
        self.init_static_info()
        self.clean_dynamic_variable()

        # TODO: key; about similarity matrix

    def init_static_info(self):
        K = self.Ndom
        self.X = np.zeros((self.Ntask, K, self.Nwork))
        self.majority_voting_result = np.zeros(self.Ntask)
        for i in range(self.Ntask):
            for j in self.NeibTask[i]:
                self.X[i, self.L[i, j] - 1, j] = 1

    def clean_dynamic_variable(self):
        K = self.Ndom
        Ndom = K
        self.A0 = np.zeros((1, self.Nwork))
        self.B0 = np.mat(np.diag([1 / float(self.v) for i in range(self.Nwork)]))
        self.probd0 = np.ones((Ndom, K, self.Nwork))  # * 0.01
        self.eta = np.dot(self.A0, self.B0.I)
        self.phi = np.zeros((Ndom, K, self.Nwork))
        self.ilm = np.ones((self.Ntask, 1))
        self.ilmc = np.zeros((self.Ntask, 1))
        self.Y = np.zeros((self.Ntask, 1)).astype(int)
        self.S = np.zeros((self.Ntask, 1)).astype(int)
        self.ans_soft_labels = np.zeros((self.Ntask, K))
        self.phic = np.zeros((Ndom, K, self.Nwork))
        self.etak = np.zeros((self.maxIter - self.burnIn, self.Nwork))
        self.etak_count = 0

    def initByMajorityVoting(self):
        if not hasattr(self, 'Y'):
            self.Y = np.zeros((self.Ntask, 1))
        for i in range(self.Ntask):
            if i == 197:
                a = 1
            ct = np.zeros((self.Ndom, 1))
            for j in range(len(ct)):
                ct[j] = sum(self.L[i, :] == j + 1)
            self.Y[i] = ct.argmax() + 1
        self.majority_voting_result = self.Y.copy()
        None

    def _train(self, ilm, eta, phi, Y, expert_instance_validation):

        K = self.Ndom
        L = self.L
        X = self.X
        c = self.c
        l = self.l
        S = self.S
        A0 = self.A0
        B0 = self.B0
        probd0 = self.probd0
        # TODO: passing similarity information here
        instances_similarity = None
        alpha2 = self.alpha2
        beta2 = self.beta2

        # variables needed to be returned
        ans_soft_labels = np.zeros(self.ans_soft_labels.shape)
        phic = np.zeros(self.phic.shape)
        etak = np.zeros(self.etak.shape)
        etak_count = 0

        for i in range(self.Ntask):
            S[i] = self.getsi(eta, i, Y[i] - 1)

        start_t = time()
        for iter in range(self.maxIter):
            A = A0.copy()
            B = B0.copy()
            for i in range(self.Ntask):
                dx = X[i, Y[i] - 1, :] - X[i, S[i] - 1, :]
                dx = dx.reshape(1, -1)
                B = B + np.dot(dx.T, dx) * ilm[i] * c * c
                A = A + (l * ilm[i] + 1 / float(c)) * dx * c * c

            eta = np.random.multivariate_normal(np.array(np.dot(A, B.I)).reshape(-1), B.I).reshape(1, -1) \
                  + self.eps

            # -- copy is necessary because probd0 should not be changed --
            probd = probd0.copy()
            for i in range(self.Ntask):
                for j in self.NeibTask[i]:
                    probd[L[i, j] - 1, Y[i] - 1, j] = probd[L[i, j] - 1, Y[i] - 1, j] + 1

            for i in range(K):
                for j in range(self.Nwork):
                    phi[:, i, j] = np.random.dirichlet(probd[:, i, j], 1) + self.eps
                    # print( "phi", phi [:,i,j])
                    # print( "probd", probd[:,i,j])

            for i in range(self.Ntask):
                dx = X[i, Y[i] - 1, :] - X[i, S[i] - 1, :]
                aczetai = abs(c * (l - np.dot(eta, dx.T)) + self.eps)
                ilm[i] = np.random.wald(np.linalg.inv(aczetai), 1)

            randomIdx = np.array(range(self.Ntask))
            np.random.shuffle(randomIdx)
            for i in randomIdx:
                logprob = np.zeros((K, 1))
                for k in range(K):
                    # if ilm[i] != 0
                    if abs(ilm[i]) > self.eps:
                        dx = X[i, k, :] - X[i, self.getsi(eta, i, k) - 1, :]
                        logprob[k] = -0.5 * ilm[i] * (1.0 / ilm[i] + c * (l - np.dot(eta, dx.T))) ** 2
                    for j in self.NeibTask[i]:
                        logprob[k] = logprob[k] + np.log(phi[L[i, j] - 1, k, j] + self.eps)

                prob = np.exp(logprob - logprob.max())
                prob = prob / prob.sum()
                prob_sample = np.random.multinomial(1, prob.reshape(-1))
                prob_nnz = np.nonzero(prob_sample > 0)
                class_k = int(prob_nnz[0]) + 1

                # instance propagation
                # if instance not be validated
                class_kind_num = len(set(expert_instance_validation[expert_instance_validation!=0]))
                if (expert_instance_validation[i] == 0):
                    labeled_idx = np.array(range(self.Ntask))[expert_instance_validation != 0]
                    if len(labeled_idx) > 0:
                        max_simi_index = instances_similarity[i, labeled_idx].argmax()
                        max_simi_index = labeled_idx[max_simi_index]
                        max_simi_value = instances_similarity[i, max_simi_index]
                        # if max_simi_value < 0.8 and class_kind_num < K:
                        #     max_simi_value = 0
                        diff_vector = (np.array(range(self.Ndom)) + 1) != \
                                      np.array(expert_instance_validation[max_simi_index]).repeat(repeats=self.Ndom)
                        loss_vector = np.abs(diff_vector)
                        prob = prob.reshape(-1) * np.exp(-beta2 * max_simi_value * loss_vector)
                        prob = prob / prob.sum()
                        prob_sample = np.random.multinomial(1, prob.reshape(-1))
                        prob_nnz = np.nonzero(prob_sample > 0)
                        class_k = int(prob_nnz[0]) + 1
                else:
                    diff_vector = (np.array(range(self.Ndom)) + 1) != \
                                  np.array(expert_instance_validation[i]).repeat(repeats=self.Ndom)
                    loss_vector = np.abs(diff_vector)
                    prob = (prob + self.eps).reshape(-1) * np.exp(-alpha2 * loss_vector)
                    prob = prob / prob.sum()
                    prob_sample = np.random.multinomial(1, prob.reshape(-1))
                    prob_nnz = np.nonzero(prob_sample > 0)
                    class_k = int(prob_nnz[0]) + 1

                Y[i] = class_k
                S[i] = self.getsi(eta, i, Y[i] - 1)

            if iter > 0:
                for i in range(self.Ntask):
                    ans_soft_labels[i, Y[i] - 1] = ans_soft_labels[i, Y[i] - 1] + 1
                phic = phic + phi
                etak[etak_count, :] = eta[0, :]
                etak_count += 1

            if self.verbose > 0:
                ans_soft_labelst = ans_soft_labels / (etak_count)
                error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,
                                                                                                  self.true_labels)
                end_t = time()
                print("iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
                # print( self.ans_soft_labels )

        ans_soft_labelst = ans_soft_labels / (etak_count)
        error_rate, soft_error_rate, error_L1, erroe_L2 = \
            self.cal_error_using_soft_label(ans_soft_labelst, self.true_labels)

        end_t = time()
        print("Final: iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
        return phic, etak_count, ans_soft_labels, Y, error_rate

    def get_majority_voting_result(self):
        self.process_data()
        self.clean_dynamic_variable()
        self.initByMajorityVoting()
        res = self.majority_voting_result.copy()
        res = np.array(res).reshape(-1) - 1
        return res

    def _train_c_buffer(self,ilm, eta, phi, Y, expert_instance_validation, worker_prior=None, simi=None, seed=None):
        # np.random.seed(24)
        if seed is None:
            seed = (int(np.random.rand()*100000))
            # seed = 4491775  # 123
        K = self.Ndom
        L = self.L.copy()
        instances_similarity = self.instances_similarity
        phi = np.zeros((K, K, self.Nwork))
        eta = np.zeros((1))
        posterior_distribution = np.zeros((self.Ntask, self.Ndom))
        if worker_prior is None:
            worker_prior = np.zeros(self.Nwork)

        if simi is None:
            modified_simi = self.instances_similarity.copy()
        else:
            modified_simi = simi.copy()

        # modified_simi = np.zeros((self.instances_similarity.shape))
        # for i in range(modified_simi.shape[0]):
        #     simi_vect = self.instances_similarity[:,i]
        #     top5 = simi_vect.argsort()[::-1][:60]
        #     # modified_simi[top5,i] = self.instances_similarity[top5,i]
        #     modified_simi[i,top5] = self.instances_similarity[top5,i]
        # modified_simi = modified_simi.transpose()
        start_t = time()
        print("begin training with dll!, seed:%s"%(seed))
        self.seed = seed
        if 1:
        # if not hasattr(self,"_data_pointer"):
            print("data preprocessing")
            self._data_pointer = init(seed, self.Ntask, K, self.Nwork, 0, self.alpha2, self.beta2,
                                      np.ascontiguousarray(L, np.float64),
                                      np.ascontiguousarray(self.true_labels, np.float64),
                                      np.ascontiguousarray(instances_similarity, np.float64))
        # incre_m3v_c(seed, self.Ntask, K, self.Nwork,
        #             L.astype(np.float64),
        #             self.true_labels.astype(np.float64),
        #             expert_instance_validation.astype(np.float64),
        #             instances_similarity.astype(np.float64),
        #             phi, eta, posterior_distribution)
        incre_m3v_c(self._data_pointer, expert_instance_validation.astype(np.float64),
                    worker_prior,
                    phi, eta, posterior_distribution)
        error_rate, soft_error_rate, error_L1, error_L2 = \
            self.cal_error_using_soft_label(posterior_distribution, self.true_labels)
        end_t = time()
        posterior_labels = posterior_distribution.argmax(axis=1)
        true_labels = np.array(self.true_labels - 1)
        s = ""
        label_list = list(set(true_labels.tolist()))
        for i in range(len(label_list)//2):
            selected_class_list = [i * 2, i * 2 + 1]
            class_indicator = [True if c in selected_class_list else False for c in true_labels]
            selected_posterior_labels = posterior_labels[class_indicator]
            selected_true_labels = true_labels[class_indicator]
            selected_sum = sum(class_indicator)
            selected_wrong = sum(selected_posterior_labels != selected_true_labels)
            s = s + str(selected_wrong) + "/" + str(selected_sum) + " "
        print(s)
        print("Final: iter:%s, error_rate:%s, acc_rate:%s, totaltime:%s" % (49, error_rate, 1-error_rate , end_t - start_t))
        return phi, eta, posterior_distribution, Y, error_rate

    def train(self, seed=None):
        self.process_data()
        self.clean_dynamic_variable()
        self.initByMajorityVoting()
        self.init_static_info()

        # get majority voting error rate
        correct_num = 0
        for i in range(self.Ntask):
            if self.Y[i] == self.true_labels[i]:
                correct_num = correct_num + 1
        mv_error_rate = 1 - float(correct_num) / float(self.Ntask)
        print("majority voting, error_rate:%s" % (mv_error_rate))

        ilm = self.ilm
        eta = self.eta
        phi = self.phi
        Y = self.Y
        worker_prior = np.zeros(self.Nwork)

        simi = np.zeros((self.Ntask, self.Ntask))

        # additional variables comparing to original m3v
        expert_instance_validation = np.zeros((self.Ntask))

        # phic, etak_count, ans_soft_labels, Y, error_rate = \
        #     self._train(ilm=ilm, eta=eta, phi=phi, Y=Y,
        #                 expert_instance_validation=expert_instance_validation)


        phic, etak_count, ans_soft_labels, Y, error_rate = \
            self._train_c_buffer(ilm=ilm, eta=eta, phi=phi, Y=Y,
                        expert_instance_validation=expert_instance_validation)

        self.phi = phic / float(etak_count)
        self.ans_soft_labelst = ans_soft_labels
        self.Y = Y
        self.error_rate = error_rate
        self.set_trained_flag()

    def set_trained_flag(self):
        self.trained = 1

    # def get_majority_voting_result(self):
    #     return self.majority_voting_result - 1


    def get_confusion_matrices(self):
        if self.trained == 0:
            print("training when trying to get confusion matrix")
            self.train()
        return self.phi

    def get_posterior_labels(self):
        if self.trained == 0:
            print("training when trying to get posterior labels")
            self.train()
        return self.ans_soft_labelst.argmax(axis=1)

    def _get_posterior_label_dist(self):
        if self.trained == 0:
            print("training when trying to get posterior label dists")
            self.train()
        return self.ans_soft_labelst

    def get_posterior_label_dist(self):
        return self._get_posterior_label_dist()

    def from_crowd_data(self, crowd_data):
        instance_num = crowd_data.get_attr("InstanceTotalNum")
        self.L = crowd_data.get_attr("WorkerLabels")
        self.true_labels = crowd_data.get_attr("true_labels")
        self.L = np.array(self.L).reshape(instance_num, -1) + 1
        self.backend_L = self.L.copy()
        self.true_labels = np.array(self.true_labels).reshape(-1) + 1
        self.origin_labels_num = (self.L > 0).sum()
        self.process_data()