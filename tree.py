import numpy as np
from tqdm import tqdm_notebook


class Tree:
    class Node:
        def __init__(self, feat, thres, val):
            self.left = None
            self.right = None
            self.val = val
            self.feat_n = feat
            self.thres = thres

        def check_cnd(self, x):
            if self.left or self.right:
                if x[self.feat_n] < self.thres:
                    return self.left.check_cnd(x)
                else:
                    return self.right.check_cnd(x)
            else:
                return self.val

    def __init__(self, max_depth, task, min_size, lmbd=1.0, gamma=0.1):
        self._max_depth = max_depth
        self.depth = 0
        self.root_node = None
        self.task = task
        self.min_size = min_size

        self.lmbd = lmbd
        self.gamma = gamma

    @staticmethod
    def __preprocess(x, y):
        if type(x) != np.ndarray:
            x = np.array(x)
        if type(y) != np.ndarray:
            y = np.array(y)
        return x, y

    '''
    select optimal split for mse error
    '''
    def __select_feature_mse(self, x, y):
        num_features = x.shape[1]
        self.value = y.mean()
        base_error = ((y - self.value) ** 2).sum()
        error = base_error
        best_feature = None
        best_thre_value = None
        best_left_mean, best_right_mean = None, None

        for feature in range(num_features):
            idxs = np.argsort(x[:, feature])
            sample_num = x.shape[0]
            right_n, left_n = sample_num, 0
            right_sum, left_sum = sum(y), 0
            right_mean, left_mean = right_sum/right_n, 0
            prev_error_right, prev_error_left = base_error, 0
            thres = 1

            while thres < sample_num - 1:
                right_n -= 1
                left_n += 1
                id = idxs[thres]

                delta_right = (right_sum - y[id])/right_n - right_mean
                delta_left = (left_sum + y[id])/left_n - left_mean

                right_sum -= y[id]
                left_sum += y[id]

                prev_error_right += (delta_right ** 2) * right_n
                prev_error_right -= (y[id] - right_mean) ** 2
                prev_error_right -= 2 * delta_right * (right_sum - right_mean * right_n)

                prev_error_left += (delta_left ** 2) * left_n
                prev_error_left += (y[id] - left_mean) ** 2
                prev_error_left -= 2 * delta_left * (left_sum - left_mean * left_n)

                right_mean = right_sum / right_n
                left_mean = left_sum / left_n

                if thres <= sample_num and np.abs(x[id, feature] - x[idxs[thres+1], feature]) < 0.000001:
                    thres += 1
                    continue

                if prev_error_left + prev_error_right < error and min(left_n, right_n) >= self.min_size:
                    error = prev_error_right + prev_error_left
                    best_feature = feature
                    best_thre_value = x[id][feature] #(x[idxs[thres - 1], best_feature] + x[id, best_feature]) / 2.0
                    best_left_mean, best_right_mean = left_mean, right_mean

                thres += 1

        return best_feature, best_thre_value, best_left_mean, best_right_mean

    def __select_feature_boosted_mse(self, x, y):
        num_features = x.shape[1]
        sample_num = x.shape[0]
        self.value = y.mean()
        best_gain = -self.gamma
        common_gain_sub = (np.sum(y)**2)/(sample_num + self.lmbd) + self.gamma
        best_feature = None
        best_thre_value = None
        best_left_mean, best_right_mean = None, None

        for feature in range(num_features):
            idxs = np.argsort(x[:, feature])
            right_n, left_n = sample_num, 0
            right_sum, left_sum = sum(y), 0
            thres = 0

            while thres < sample_num - 1:
                id = idxs[thres]
                right_n -= 1
                left_n += 1
                right_sum -= y[id]
                left_sum += y[id]

                g_l = left_sum
                g_r = right_sum

                gain = (g_l ** 2)/(left_n + self.lmbd) + (g_r ** 2)/(right_n + self.lmbd) - common_gain_sub

                if thres < sample_num-1 and np.abs(x[id, feature] - x[idxs[thres+1], feature]) < 0.000001:
                    thres += 1
                    continue

                if gain > best_gain and gain > 0 and min(left_n, right_n) >= self.min_size:
                    best_gain = gain
                    best_feature = feature
                    best_thre_value = x[id][feature] #(x[idxs[thres], best_feature] + x[idxs[thres+1], best_feature]) / 2
                    best_left_mean = -(left_sum / (left_n+self.lmbd))
                    best_right_mean = -(right_sum / (right_n+self.lmbd))

                thres += 1
        return best_feature, best_thre_value, best_left_mean, best_right_mean

    def __recursion_build(self, x, y, lvl, val):
        if lvl + 1 <= self._max_depth:
            if self.task == 'mse':
                select_feature = self.__select_feature_mse
            else:  # self.task == 'xgboost_mse':
                select_feature = self.__select_feature_boosted_mse
            feat, thres_value, left_mean, right_mean = select_feature(x, y)
            new_node = Tree.Node(feat, thres_value, val)
            if feat is not None and thres_value is not None:
                il, ir = x[:, feat] < thres_value, x[:, feat] >= thres_value
                if len(y[ir]) >= self.min_size and len(y[il]) >= self.min_size:
                    new_node.left = self.__recursion_build(x[il, :], y[il], lvl+1, left_mean)
                    new_node.right = self.__recursion_build(x[ir, :], y[ir], lvl+1, right_mean)
        else:
            new_node = Tree.Node(None, None, val)
        return new_node

    def fit(self, x, y):
        x, y = Tree.__preprocess(x, y)
        self.root_node = self.__recursion_build(x, y, 0, y.mean())

    def __predict_one(self, x):
        return self.root_node.check_cnd(x)

    def predict(self, x):
        return np.array([self.__predict_one(x_i) for x_i in x])


class TreeBagging:
    def __init__(self, n_models, task, max_depth, min_size, data_subsample, feature_subsample=None, lmbd=1.0, gamma=0.1):
        self.n_models = n_models
        self.task = task
        self.max_depth = max_depth
        self.min_size = min_size
        self.data_subsample = data_subsample
        self.feature_subsample = feature_subsample
        self.lmbd = lmbd
        self.gamma = gamma
        self.ensamble = []

    @staticmethod
    def __preprocess(x, y):
        if type(x) != np.ndarray:
            x = np.array(x)
        if type(y) != np.ndarray:
            y = np.array(y)
        return x, y

    def fit(self, x, y, replace=False):
        x, y = TreeBagging.__preprocess(x, y)
        self.ensamble = []
        for i in range(self.n_models):
            n_sample = int(x.shape[0] * self.data_subsample)
            sub_idx = np.random.choice(x.shape[0], n_sample, replace=replace)
            new_model = Tree(max_depth=self.max_depth, task=self.task, min_size=self.min_size)
            new_model.fit(x[sub_idx], y[sub_idx])
            self.ensamble.append(new_model)


    def predict(self, x):
        pred = []
        for model in self.ensamble:
            pred.append(model.predict(x))
        pred = np.array(pred).T
        return np.array([np.mean(line) for line in pred])


class TreeGradientBoosting:
    def __init__(self, n_models, task, max_depth, min_size=5, data_subsample=1.0, learning_rate=0.9, learning_rate_dev=1.5,
                 feature_subsample=None, add_max_depth=0, no_change_val=1.5, no_change_dev=1.5, lmbd=1.0, gamma=0.1, init_mse=True):
        self.n_models = n_models
        self.task = task
        self.max_depth = max_depth
        self.min_size = min_size
        self.lmbd = lmbd
        self.gamma = gamma
        self.data_subsample = data_subsample
        self.feature_subsample = feature_subsample
        self.learning_rate = learning_rate

        self.add_max_depth = add_max_depth
        self.learning_rate_dev = learning_rate_dev
        self.no_change_val = no_change_val
        self.no_change_dev = no_change_dev
        self.init_mse = init_mse

        self.ensamble = []
        self.lrs = []
        self.stat = []
        self.mean = 0.0


    @staticmethod
    def __preprocess(x, y):
        if type(x) != np.ndarray:
            x = np.array(x)
        if type(y) != np.ndarray:
            y = np.array(y)
        return x, y

#todo add classification
    def fit(self, x, y, replace=False):
        x, y = TreeGradientBoosting.__preprocess(x, y)
        self.ensamble = []
        self.lrs = []
        self.mean = np.mean(y)
        max_depth = self.max_depth
        learning_rate = self.learning_rate
        no_change_val = self.no_change_val

        y_target = y

        #is any effect
        if self.init_mse:
            y_pred = np.ones([y.shape[0]])*self.mean
        else:
            y_pred = np.zeros([y.shape[0]])

        for i in tqdm_notebook(range(self.n_models)):
            n_sample = int(x.shape[0] * self.data_subsample)
            sub_idx = np.random.choice(x.shape[0], n_sample, replace=replace)

            new_model = Tree(max_depth=max_depth, task=self.task, min_size=self.min_size,
                             lmbd=self.lmbd, gamma=self.gamma)
            if self.task == 'xgboost_mse':
                new_model.fit(x[sub_idx], -y_target[sub_idx])
            else:
                new_model.fit(x[sub_idx], y_target[sub_idx])
            self.ensamble.append(new_model)
            self.lrs.append(learning_rate)

            y_pred += np.array(new_model.predict(x)) * learning_rate
            prev_target = y_target
            y_target = y-y_pred

            if (np.sum(prev_target - y_target) ** 2) / y.shape[0] < no_change_val:
                no_change_val /= self.no_change_dev
                learning_rate /= self.learning_rate_dev
                max_depth += self.add_max_depth

            if i > 0:
                self.stat.append((y_target**2 / y.shape[0]))

    def predict(self, x):
        if self.init_mse:
            y = np.ones([x.shape[0]]) * self.mean
        else:
            y = np.zeros(x.shape[0])
        for i in range(self.n_models):
            y += self.ensamble[i].predict(x) * self.lrs[i]
        return y
