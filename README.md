# Pytorch를 활용한 강화학습/심층강화학습 실전 입문 도서 연습
[코드 참고] : https://github.com/wikibook/pytorch-drl/tree/master/program


### 파이토치에서 클래스로 모델을 개발할때 알아두면 좋은 정보
1. 신경망 클래스를 구현할 때 `super().__init__()`함수를 호출하고 layer를 만든 뒤 forward메소드를 통해 각 layer에서 통과할 activation함수를 결정한다. 이때 어떻게 모델이 호출될까? </br>
아래와 같은 신경망이 존재한다고 하자 </br>

<pre>
<code>
class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out) # 행동을 결정하는 부분이므로 출력개수는 행동의 가짓수
        self.critic = nn.Linear(n_mid, 1) # 상태 가치를 출력하는 부분이므로 출력개수는 1개
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2) # 상태가치 계산
        actor_output = self.actor(h2) # 행동가치 계산
</code>
</pre>
- 여기서 우리가 주목해야 할 곳은 `super(Net, self).__init__()`이다.
- 위 함수가 호출되어 nn.Module에서 `__init__()`을 실행하면 `__call__` 메소드가 호출되어 모델에 전달된 값을 호출하여 forward함수에 인자를 전달함으로써 신경망을 통과하게 된다.
