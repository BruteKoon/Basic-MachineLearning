# 텐서 플로우 기본적인 구성 익히기
import tensorflow as tf

#tf.constant : 상수
hello = tf.constant('HEllo, Tensorflow!')
print (hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf. add(a,b)
print(c)


#위에서 변수와 수식들을 정의했지만, 실행을 정의한 시점에서 실행되는것 아님
#다음처럼 Session 객체와 run 메소드를 사용할 때 계산
# 모델을 구성하는 것과, 실행하는 것을 분리하여 프로그램을 깔끔하게 작성 가능

#그래프를 실행할 세션 구성
sess = tf.Session()
#sess.run : 설정한 텐서 그래프를 실행
print(sess.run(hello))
print(sess.run([a,b,c]))

#세션 닫기
sess.close()
