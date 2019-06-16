
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
random_state = 42

def tf_log(x):
    # WRITE ME

### ネットワーク ###
tf.reset_default_graph()
is_training = tf.placeholder(tf.bool, shape=())

# WRITE ME

y = # WRITE ME

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

### 前処理 ###
def gcn(x):
    # WRITE ME

class ZCAWhitening:
    # WRITE ME
    
x_train, x_test, t_train = load_cifar10()
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.1, random_state=random_state)
zca = ZCAWhitening()
zca.fit(x_train)
x_train_zca = zca.transform(gcn(x_train))
t_train_zca = t_train[:]
x_valid_zca = zca.transform(gcn(x_valid))
t_valid_zca = t_valid[:]
x_test_zca = zca.transform(gcn(x_test))

### 学習 ###
n_epochs = 10
batch_size = 100
n_batches = x_train.shape[0]//batch_size

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(n_epochs):
    # WRITE ME
sess.close()

y_pred = # WRITE ME
submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission_pred.csv', header=True, index_label='id')