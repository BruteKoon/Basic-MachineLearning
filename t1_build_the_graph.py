import tensorflow as tf

# 1x2 행렬을 만드는 constant op을 만들어 봅시다.
# 이 op는 default graph에 노드로 들어갈 것입니다.
# Create a constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.

# 생성함수에서 나온 값은 constant op의 결과값입니다.
# The value returned by the constructor represents the output
# of the constant op.

matrix1 = tf.constant([[3.,3.]])

# 2x1 행렬을 만드는 constant op을 만들어봅시다.
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2를 입력값으로 하는 Matmul op(역자 주: 행렬곱 op)을
# 만들어 봅시다.
# 이 op의 결과값인 'product'는 행렬곱의 결과를 의미합니다.
# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)


# default graph를 실행시켜 봅시다.
# Launch the default graph.
sess = tf.Session()

# 행렬곱 작업(op)을 실행하기 위해 session의 'run()' 메서드를 호출해서 행렬곱 
# 작업의 결과값인 'product' 값을 넘겨줍시다. 그 결과값을 원한다는 뜻입니다.
# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# 작업에 필요한 모든 입력값들은 자동적으로 session에서 실행되며 보통은 병렬로 
# 처리됩니다.
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# 'run(product)'가 호출되면 op 3개가 실행됩니다. 2개는 상수고 1개는 행렬곱이죠.
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# 작업의 결과물은 numpy `ndarray` 오브젝트인 result' 값으로 나옵니다.
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)

# 실행을 마치면 Session을 닫읍시다.
# Close the Session when we're done.
sess.close()


