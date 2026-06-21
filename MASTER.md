# RL-Mechanical-Engineering

기계공학(Mechanical Engineering)과 강화학습(Reinforcement Learning)을 결합하여 실제 물리 시스템을 모델링하고, AI를 이용해 제어 및 최적화하는 것을 목표로 한다.

---

# Long-Term Goal

최종적으로는 다음 중 하나 이상의 프로젝트를 완성한다.

* RL Vehicle Dynamics
* RL Structural Optimization
* RL Thermal Control
* RL Suspension Design
* RL Robotics
* RL Control System

단순히 강화학습을 사용하는 것이 아니라,

기계공학적 지식을 활용하여

Physics → Simulation → RL → Optimization

의 전체 과정을 이해하는 것을 목표로 한다.

---
# Project Rules

## Rule 1

모든 Project 완료 후 MASTER.md에 기록한다.

---
## Rule 2

코드보다 사고 과정을 기록한다.

기록할 내용:

* 왜 이런 설계를 선택했는가?
* 어떤 실수를 했는가?
* 무엇을 배웠는가?
* 다음 단계는 무엇인가?

---

## Rule 3

모든 프로젝트는 아래 형식을 따른다.

### 목표

### 배운 내용

### 결과

### 느낀 점

### 다음 목표

---

## Rule 4

코드가 실행되는 것만으로 완료가 아니다.

아래 개념을 설명할 수 있어야 완료로 인정한다.

* State
* Action
* Reward
* Physics
* 알고리즘 핵심 아이디어

---

## Rule 5

새로운 프로젝트를 시작하기 전, 이전 프로젝트 내용을 1분 안에 설명할 수 있어야 한다.

---

# Environment Setup

Hardware

* MacBook Pro M4 Pro
* Windows Desktop (RTX 3070)

개발 환경

* Miniconda
* Python 3.11
* VS Code
* Git

주요 라이브러리

* NumPy
* Matplotlib
* Gymnasium
* Stable-Baselines3
* PyTorch

---

# Learning Roadmap

Level 0

Physics Simulation Basics

* Falling Ball
* Car Simulator

---

Level 1

RL Environment Design

* State
* Action
* Reward
* Transition

---

Level 2

Gymnasium Environment

* Custom Car Environment
* Random Agent
* PPO Training

---

Level 3

Mechanical Engineering Integration

* Vehicle Dynamics
* Thermal Systems
* Structural Mechanics
* Control Systems

---

# Completed Projects

---

# 2026-06-16 — Project 01: Falling Ball Simulator

## 목표

Euler Method를 이용하여 자유낙하 운동을 시뮬레이션한다.

---

## 배운 내용

### Euler Method

속도 업데이트

v = v + a * dt

위치 업데이트

x = x + v * dt

를 사용하여 물체의 운동을 계산하였다.

---

### 수치해석

컴퓨터는

v = v0 + at

같은 해석해를 사용하는 것이 아니라,

작은 시간 간격(dt)마다 상태를 갱신하며 운동을 계산한다.

---

### State

공의 상태는

State = [y, v]

로 표현 가능하다.

---

### dt

dt는 시뮬레이션의 시간 간격이다.

dt가 작을수록 정확도가 높아지지만 계산량이 증가한다.

---

## 결과

100m 높이에서 자유낙하하는 공의 운동을 시뮬레이션하였다.

높이-시간 그래프가 포물선 형태로 감소하는 것을 확인하였다.

---

## 느낀 점

강화학습 이전에 Physics Environment를 직접 구현하는 것이 중요하다는 것을 이해하였다.

---

## 다음 목표

Project 02: Car Simulator

---

# 2026-06-16 — Project 02: 1D Car Simulator

## 목표

자동차의 위치와 속도를 시뮬레이션한다.

---

## 배운 내용

### Newton's Second Law

F = ma

를 이용하여 가속도를 계산하였다.

a = F / m

---

### Velocity Update

속도는

v = v + a * dt

로 계산된다.

---

### Position Update

위치는

x = x + v * dt

로 계산된다.

---

### State Design

자동차의 상태는

State = [x, v]

로 정의하였다.

---

### Markov Property

같은 위치에 있더라도 속도가 다르면 미래가 달라진다.

예:

x = 50, v = +20

x = 50, v = -20

따라서 상태에는 속도 정보가 반드시 포함되어야 한다.

---

### Reward Design Discussion

보상 설계에 대해 토론하였다.

A

목표 도달 시만 보상

B

목표에 가까워질수록 보상

C

목표에 정확히 도달하면 큰 보상, 초과 시 페널티

Dense Reward(B)가 초기 학습에 유리할 수 있다는 아이디어를 도출하였다.

또한 학습 초기에 B를 사용하고 이후 C로 전환하는 Curriculum Learning과 유사한 접근을 논의하였다.

---

## 결과

속도 그래프는 직선 형태,

위치 그래프는 포물선 형태로 나타남을 확인하였다.

---

## 느낀 점

강화학습의 핵심은 알고리즘 이전에

State

Action

Reward

를 설계하는 것이라는 점을 이해하였다.

---

## 다음 목표

Project 03: Car With Actions

Action → Force → Physics → State

구조를 구현한다.

# 2026-06-16 — Project 03: Car With Actions

## 목표

자동차가 Action에 따라 가속, 유지, 감속할 수 있는 시뮬레이션을 구현한다.

---

## 배운 내용

### Action 개념

자동차는 더 이상 고정된 힘을 받지 않는다.

Agent가 선택한 Action에 따라 힘이 결정된다.

예:

Action = -1 → Brake

Action = 0 → Coast

Action = +1 → Accelerate

---

### Action → Force Mapping

Action을 실제 물리량으로 변환하였다.

F = action × force_scale

를 사용하여 행동이 물리 시스템에 영향을 주도록 구현하였다.

---

### Transition Function

현재 상태와 행동으로부터 다음 상태를 계산하였다.

State = [x, v]

Action = a

↓

Force = action × force_scale

↓

Acceleration = F / m

↓

Velocity Update

↓

Position Update

↓

Next State = [x', v']

즉,

(State, Action) → Next State

구조를 처음 구현하였다.

---

### Time Tracking

초기 구현에서는

times.append(action * dt)

를 사용하여 시간과 행동을 혼동하였다.

수정 후

times.append(step * dt)

를 사용하여 시간축을 올바르게 기록하였다.

---

### Physics Engine 관점

현재까지 구현한 시스템은 RL Agent가 아니라 Physics Engine이다.

Physics Engine의 역할은

현재 상태와 행동으로부터

다음 상태를 계산하는 것이다.

---

### State Space와 제어공학

현재 시스템은

State = [x, v]

로 표현된다.

향후에는

[x, y, vx, vy]

또는

[x, y, z, vx, vy, vz]

와 같은 고차원 상태공간으로 확장 가능함을 이해하였다.

또한

s(k+1) = A s(k) + B u(k)

형태의 상태방정식과 RL Environment의 관계를 논의하였다.

---

## 결과

Action Sequence

[1, 1, 1, 0, -1, -1]

를 적용하여

Force

Acceleration

Velocity

Position

그래프를 생성하였다.

행동에 따라 자동차가 가속, 유지, 감속하는 것을 확인하였다.

---

## 느낀 점

강화학습은 단순히 신경망을 학습시키는 것이 아니라,

State

Action

Reward

Physics

Transition

을 설계하는 과정이라는 점을 더욱 명확히 이해하였다.

또한 RL과 제어공학이 생각보다 밀접하게 연결되어 있다는 것을 알게 되었다.

---

## 다음 목표

Project 04: CarEnv

Gymnasium 스타일의 Environment를 직접 구현한다.

env.reset()

env.step(action)

구조를 설계한다.
# 2026-06-16 — Project 03: Car With Actions

## 목표

자동차가 Action에 따라 가속, 유지, 감속할 수 있는 시뮬레이션을 구현한다.

---

## 배운 내용

### Action 개념

자동차는 더 이상 고정된 힘을 받지 않는다.

Agent가 선택한 Action에 따라 힘이 결정된다.

예:

Action = -1 → Brake

Action = 0 → Coast

Action = +1 → Accelerate

---

### Action → Force Mapping

Action을 실제 물리량으로 변환하였다.

F = action × force_scale

를 사용하여 행동이 물리 시스템에 영향을 주도록 구현하였다.

---

### Transition Function

현재 상태와 행동으로부터 다음 상태를 계산하였다.

State = [x, v]

Action = a

↓

Force = action × force_scale

↓

Acceleration = F / m

↓

Velocity Update

↓

Position Update

↓

Next State = [x', v']

즉,

(State, Action) → Next State

구조를 처음 구현하였다.

---

### Time Tracking

초기 구현에서는

times.append(action * dt)

를 사용하여 시간과 행동을 혼동하였다.

수정 후

times.append(step * dt)

를 사용하여 시간축을 올바르게 기록하였다.

---

### Physics Engine 관점

현재까지 구현한 시스템은 RL Agent가 아니라 Physics Engine이다.

Physics Engine의 역할은

현재 상태와 행동으로부터

다음 상태를 계산하는 것이다.

---

### State Space와 제어공학

현재 시스템은

State = [x, v]

로 표현된다.

향후에는

[x, y, vx, vy]

또는

[x, y, z, vx, vy, vz]

와 같은 고차원 상태공간으로 확장 가능함을 이해하였다.

또한

s(k+1) = A s(k) + B u(k)

형태의 상태방정식과 RL Environment의 관계를 논의하였다.

---

## 결과

Action Sequence

[1, 1, 1, 0, -1, -1]

를 적용하여

Force

Acceleration

Velocity

Position

그래프를 생성하였다.

행동에 따라 자동차가 가속, 유지, 감속하는 것을 확인하였다.

---

## 느낀 점

강화학습은 단순히 신경망을 학습시키는 것이 아니라,

State

Action

Reward

Physics

Transition

을 설계하는 과정이라는 점을 더욱 명확히 이해하였다.

또한 RL과 제어공학이 생각보다 밀접하게 연결되어 있다는 것을 알게 되었다.

---

## 다음 목표

Project 04: CarEnv

Gymnasium 스타일의 Environment를 직접 구현한다.

env.reset()

env.step(action)

구조를 설계한다.

# 2026-06-16 — Project 04: CarEnv

## 목표

기존의 물리 시뮬레이터를 강화학습(Environment) 형태로 변환한다.

Gymnasium의 핵심 구조인

* reset()
* step(action)

을 직접 구현한다.

---

## 배운 내용

### Environment 설계

기존에는 단순한 시뮬레이션 코드였다.

이번 프로젝트에서는

```python
env = CarEnv()
```

형태의 환경(Environment)을 직접 구현하였다.

---

### State

환경의 상태(State)를

```python
[x, v]
```

로 정의하였다.

* x : 위치
* v : 속도

현재 상태만 알면 다음 상태를 계산할 수 있음을 이해하였다.

---

### Encapsulation

전역 변수 대신

```python
self.x
self.v
```

를 사용하였다.

이를 통해 환경 내부 상태를 클래스가 관리하도록 구현하였다.

---

### reset()

```python
state = env.reset()
```

를 구현하였다.

역할:

* 위치 초기화
* 속도 초기화
* Episode 시작

반환값:

```python
[x, v]
```

---

### step(action)

```python
next_state, reward, done = env.step(action)
```

구조를 구현하였다.

계산 과정:

```text
Action
↓
Force
↓
Acceleration
↓
Velocity Update
↓
Position Update
↓
Next State
```

---

### Physics Engine

물리 모델:

```python
F = action * force_scale

a = F / m

v = v + a * dt

x = x + v * dt
```

을 사용하였다.

이를 통해 행동(Action)이 실제 물리 상태에 영향을 주도록 구현하였다.

---

### Reward Function

초기 Reward는

```python
reward = -abs(target - x)
```

를 사용하였다.

---

#### 발견한 문제점

현재 Reward는

```text
x = 9m
reward = -1

x = 11m
reward = -1
```

과 같이

목표 지점을 지나쳤는지 여부를 구분하지 못한다.

---

#### 논의

더 자연스러운 Reward는

```text
목표에 가까워질수록 +

목표에서 멀어질수록 -
```

방식일 수 있다는 점을 발견하였다.

향후

Distance Reduction Reward

를 실험하기로 하였다.

---

### Policy 와 Environment 분리

현재 Policy는

```python
action = 1
```

이다.

즉

항상 가속하는 정책이다.

---

중요한 점:

```text
Policy
=
행동 결정

Environment
=
물리 계산
```

임을 이해하였다.

---

### Episode 와 Epoch

머신러닝의 Epoch와 달리

강화학습에서는

```text
Episode
=
시작부터 종료까지의 한 게임
```

이라는 개념을 사용함을 학습하였다.

---

### Deterministic Environment

현재 환경은

```text
같은 State
+
같은 Action
=
항상 같은 Next State
```

를 생성한다.

즉

Deterministic Environment

임을 이해하였다.

---

### 현실과의 차이

실제 시스템에는

* 센서 노이즈
* 마찰 변화
* 외란

등이 존재한다.

향후

```python
np.random.normal(...)
```

을 활용한 확률적(Stochastic) 환경으로 확장 가능함을 논의하였다.

---

### VS Code Debugging

처음으로 VS Code의 디버거를 사용하였다.

확인한 변수:

* state
* action
* reward
* done
* next_state

디버거를 통해 상태 갱신(State Update)을 직접 추적하였다.

---

## 결과

CarEnv 클래스를 구현하였다.

다음 코드가 정상 동작함을 확인하였다.

```python
env = CarEnv()

state = env.reset()

for step in range(10):

    action = 1

    next_state, reward, done = env.step(action)

    print(step, next_state, reward, done)
```

---

## 느낀 점

강화학습은 단순히 신경망을 학습시키는 것이 아니라

* State
* Action
* Reward
* Transition
* Environment

를 설계하는 과정이라는 점을 이해하였다.

또한 Reward 설계가 학습 결과에 매우 큰 영향을 준다는 사실을 체감하였다.

---

## 다음 목표

Project 05: Random Policy

* 랜덤 행동 선택
* 여러 Reward 설계 비교
* Trajectory 시각화
* Policy의 역할 체험

# 2026-06-17 — Project 05: Random Policy

## 목표

Random Policy가 Environment에서 어떤 행동을 보이는지 관찰한다.

---

## 배운 내용

### Random Policy

기존의

```python
action = 1
```

고정 정책 대신

```python
action = random.choice([-1, 0, 1])
```

를 사용하였다.

이를 통해 Policy가 행동 선택 규칙임을 이해하였다.

---

### Trajectory

매 step마다

* Position
* Velocity
* Action
* Reward

를 기록하였다.

Trajectory는 상태(State)의 연속적인 변화 과정임을 학습하였다.

---

### 데이터 기록

다음 데이터를 저장하였다.

```python
times
positions
velocities
actions
rewards
```

이를 통해 단순 print 대신 전체 Episode를 분석할 수 있게 되었다.

---

### 시각화

다음 그래프를 작성하였다.

* Position vs Time
* Velocity vs Time
* Action vs Time
* Reward vs Time

그래프를 통해 정책이 Environment에 미치는 영향을 확인하였다.

---

### Reward 분석

현재 Reward 함수

```python
reward = -abs(target - x)
```

를 사용하였다.

실험 과정에서 다음 문제를 발견하였다.

```text
x = 9m  → reward = -1
x = 11m → reward = -1
```

즉 목표를 지나쳤는지 여부를 구분하지 못한다.

Reward 설계가 학습 결과에 매우 중요함을 이해하였다.

---

### Random Policy의 한계

랜덤 정책은 목표 지점에 도달할 수도 있지만,

* 불필요한 가속
* 불필요한 감속
* 후진

등의 행동을 수행한다.

따라서 안정적으로 목표를 달성하지 못한다.

---

### Deterministic Environment

같은 상태(State)와 행동(Action)이 주어지면 항상 같은 결과가 나온다.

현재 Environment는 Deterministic Environment임을 이해하였다.

또한 현실에서는 노이즈와 외란이 존재하므로 향후 확률적(Stochastic) 환경으로 확장 가능함을 논의하였다.

---

### Debugging

VS Code Debugger를 처음 사용하였다.

브레이크포인트를 통해

* state
* action
* reward
* next_state

값을 직접 확인하였다.

State Update 과정을 단계별로 추적할 수 있음을 학습하였다.

---

## 결과

Random Policy를 구현하였다.

Policy에 따라 Trajectory가 달라짐을 확인하였다.

또한 Reward는 행동을 평가하는 기준이며, Policy는 행동을 선택하는 규칙임을 구분할 수 있게 되었다.

---

## 다음 목표

Project 06: Rule-Based Policy

Random Policy 대신 사람이 직접 만든 정책을 구현한다.

예시:

```python
if x < target:
    action = 1
else:
    action = -1
```

Random Policy와 Rule-Based Policy를 비교하여 Policy의 중요성을 체험한다.

# 2026-06-17 — Project 06: Rule-Based Policy

## 목표

Random Policy 대신 사람이 직접 설계한 Rule-Based Policy를 구현한다.

---

## 배운 내용

### Rule-Based Policy

다음 정책을 구현하였다.

```python
if x < target:
    action = 1
else:
    action = -1
```

위치(x)를 기준으로 행동을 결정하였다.

---

### Policy와 State의 관계

Environment의 상태(State)는

```python
[x, v]
```

로 구성되어 있었다.

하지만 이번 Policy는

```python
x
```

만 사용하고

```python
v
```

는 사용하지 않았다.

---

### Overshoot 발생

실험 결과 목표 위치인

```text
10 m
```

근처에서 멈추지 못하고

```text
10 m
↓
20 m
↓
0 m
```

수준의 큰 진동이 발생하였다.

---

### 원인 분석

현재 Policy는

```text
위치(x)는 고려

속도(v)는 무시
```

하였다.

따라서

```text
x = 9.9 m

v = 매우 큰 양의 값
```

인 경우에도

계속 가속을 선택하게 된다.

결과적으로 목표를 크게 지나치는 현상이 발생하였다.

---

### State 활용의 중요성

좋은 State를 정의하는 것과

State 정보를 실제 Policy에서 활용하는 것은 다른 문제임을 이해하였다.

이번 실험을 통해

```text
State 설계

↓

Policy 설계
```

가 모두 중요함을 확인하였다.

---

### Control 관점

이번 결과는 단순한 RL 문제가 아니라

제어공학(Control)의 관점에서도 해석할 수 있다.

현재 정책은

```text
풀가속

또는

풀브레이크
```

만 존재하는 Bang-Bang Control 형태이다.

이 때문에 큰 Overshoot와 진동이 발생하였다.

---

### Episode 종료 조건

초기에는

```python
done = x >= target
```

조건 때문에 목표 지점 도달 즉시 Episode가 종료되었다.

따라서 진동 현상을 관찰할 수 없었다.

이후 종료 조건을 제거하여 Policy의 실제 동작을 확인하였다.

---

### Debugging

VS Code Debugger를 사용하여

* state
* action
* reward
* next_state

를 추적하였다.

State Update 과정을 직접 확인하였다.

---

## 결과

Rule-Based Policy는 Random Policy보다 목적성이 있었지만,

속도 정보를 고려하지 않아 안정적으로 목표 위치에 도달하지 못하였다.

---

## 핵심 교훈

좋은 Policy는

```text
위치만 보는 것
```

이 아니라

```text
위치 + 속도
```

를 함께 고려해야 한다.

---

## 다음 목표

Project 07: Better Rule-Based Policy

속도(v)를 활용하여 Overshoot를 줄이는 정책을 설계한다.

# 2026-06-17 — Project 07: Better Rule-Based Policy

## 목표

Rule-Based Policy를 개선하여 목표 위치 근처에서 안정적으로 정지하도록 만든다.

---

## 배운 내용

### 기존 정책의 문제점

Project 06의 정책

```python
if x < target:
    action = 1
else:
    action = -1
```

은 위치(x)만 고려하였다.

---

결과적으로

```text
목표 지점을 크게 지나침

(Overshoot)

↓

큰 진동 발생
```

하였다.

---

### 속도(v)의 중요성

상태(State)는

```python
[x, v]
```

였지만

기존 정책은

```python
x
```

만 사용하였다.

---

실험을 통해

```text
좋은 Policy는

위치뿐 아니라

속도도 고려해야 한다.
```

는 사실을 확인하였다.

---

### Stopping Distance

처음으로

```text
미래를 예측하는 정책
```

을 설계하였다.

---

사용한 공식

```text
Stopping Distance

= v² / (2a)
```

---

의미

```text
현재 속도로 달리는 자동차가

최대 제동을 했을 때

멈추기까지 필요한 거리
```

이다.

---

### Stopping Distance 유도

사용한 운동학 식

```text
v² = v₀² + 2aΔx
```

---

정지 조건

```text
v = 0
```

을 대입하면

```text
Δx = v₀² / (2|a|)
```

를 얻는다.

---

### 미래 상태 예측

기존 정책

```text
현재 위치만 확인
```

---

개선 정책

```text
현재 속도

↓

멈추는 데 필요한 거리 계산

↓

브레이크 여부 결정
```

---

즉

```text
현재 상태

↓

미래 결과 예측

↓

행동 선택
```

구조로 발전하였다.

---

### Policy 설계

핵심 아이디어

```python
if stopping_distance >= error:
    action = -1
```

---

의미

```text
지금 브레이크를 밟지 않으면

목표를 지나치게 된다.
```

---

### 제어(Control) 관점

이번 프로젝트를 통해

단순한 위치 제어가 아닌

```text
위치 + 속도
```

를 함께 고려하는 제어의 필요성을 이해하였다.

---

## 핵심 교훈

좋은 Policy는

```text
현재 상태만 보는 것이 아니라

현재 상태로부터

미래 결과를 예측해야 한다.
```

---

## 결과

Project 06보다 훨씬 안정적인 Policy를 설계하였다.

또한

```text
Overshoot

Stopping Distance

Prediction
```

개념을 이해하였다.

---

## 다음 목표

Project 08: Reward Engineering

Reward 함수의 설계가 Policy 학습에 미치는 영향을 분석한다.


# 2026-06-18 — Project 08: Reward Engineering

## 목표

Reward 함수를 비교하고 Reward의 역할을 이해한다.

---

## 배운 내용

### Reward A

reward = -abs(target - x)

현재 상태가 목표와 얼마나 가까운지를 평가한다.

---

### Reward B

reward = previous_distance - current_distance

이전 상태 대비 목표에 가까워졌는지 또는 멀어졌는지를 평가한다.

---

### Reward의 역할

Reward는 행동(Action)을 결정하지 않는다.

Reward는 행동의 결과를 평가하는 점수(score)이다.

---

### Reward와 Policy의 차이

Policy는

State → Action

을 결정한다.

Reward는

Action의 결과가 얼마나 좋았는지 평가한다.

---

### 중요한 발견

Reward A와 Reward B를 사용해도 Position, Velocity, Action 그래프는 거의 동일하였다.

그 이유는 현재 시스템이 Rule-Based Policy를 사용하고 있기 때문이다.

Reward는 계산되지만 행동 결정에 사용되지 않는다.

---

### Reward Engineering의 핵심

좋은 Reward는 Agent에게

"무엇이 좋은 행동인가"

를 명확하게 알려준다.

Reward B는 목표에 가까워지는 행동에 즉시 양의 보상을 제공하므로 행동의 품질을 더 직접적으로 평가할 수 있다.

---

## 결과

Reward A와 Reward B를 구현하고 비교하였다.

Reward 함수가 Trajectory를 만드는 것이 아니라 Trajectory를 평가한다는 점을 이해하였다.

---

## 핵심 교훈

Reward는 행동을 결정하지 않는다.

Reward는 행동을 평가한다.

Reward가 행동에 영향을 주려면 Q-Learning과 같은 학습 알고리즘이 필요하다.

---

## 다음 목표

Project 09: Q-Learning

Reward를 이용하여 실제로 Policy를 학습시키는 알고리즘을 구현한다.


Project 09 (Concept)

Q-Learning의 핵심 개념 이해

- Reward vs Q-value
- Gamma Discount
- Bellman Error
- Learning Rate
- Exploration vs Exploitation
- Epsilon-Greedy
- Continuous State Problem
- State Discretization
- Q-Table Structure

구현 전 개념 학습 완료

# Project 09 - Q-Learning

## 목표

Reward를 단순히 기록하는 것이 아니라 실제 행동(Action)에 영향을 주도록 만드는 첫 번째 강화학습 알고리즘 구현.

---

## 학습한 개념

### State

현재 환경의 상태.

CarEnv에서는

* x (위치)
* v (속도)

로 구성된다.

```python
state = [x, v]
```

---

### Action

에이전트가 선택하는 행동.

```python
actions = [-1, 0, 1]
```

* -1 : 브레이크
* 0 : 유지
* 1 : 가속

---

### Reward

행동 결과에 대한 채점.

Reward B 사용.

```text
목표와 가까워지면 +1
목표에서 멀어지면 -1
```

---

### Q(s,a)

상태 s에서 행동 a를 했을 때의 가치.

정확히는

```text
현재 보상 + 미래 보상
```

까지 포함한 기대 가치.

---

### V(s)

상태 자체의 가치.

최적 정책에서는

V(s) = max_a Q(s,a)

즉,

"이 상태에 도착했을 때 가장 좋은 행동을 하면 얼마나 좋은가?"

를 의미한다.

---

### Bellman Equation

Q-Learning의 핵심.

```text
Q(s,a)

=

reward

+

gamma * max_a' Q(s',a')
```

현재 행동의 가치는

* 지금 받은 보상
* 다음 상태의 미래 가치

를 합친 값으로 정의된다.

---

### Bellman Error

현재 추정값과 정답(Target)의 차이.

```text
error

=

target

-

Q(s,a)
```

---

### Learning Rate (alpha)

Q값을 얼마나 빠르게 수정할지 결정.

```python
alpha = 0.1
```

---

### Discount Factor (gamma)

미래 보상의 중요도.

```python
gamma = 0.95
```

gamma가 클수록 미래를 중요하게 생각한다.

---

### Exploration vs Exploitation

Exploration

```text
새로운 행동 시도
```

Exploitation

```text
현재 가장 좋은 행동 선택
```

---

### Epsilon-Greedy

```python
epsilon = 0.1
```

90%

```text
현재 최선 행동
```

10%

```text
랜덤 행동
```

---

## State Discretization

연속 상태를 Q-Table에 저장하기 위해 사용.

```python
def discretize_state(state):
    x_bin = int(state[0])
    v_bin = int(state[1])

    return (x_bin, v_bin)
```

예시

```text
(3.42, 1.89)

↓

(3, 1)
```

---

## Q-Table

Dictionary 기반 구현.

```python
Q = {}
```

구조

```python
Q[(3,1)] = [2.0, 5.0, 1.0]
```

의미

```text
action = -1 → 2.0
action =  0 → 5.0
action =  1 → 1.0
```

---

## 구현한 함수

### discretize_state()

상태를 bin으로 변환.

### get_q_values()

새 상태 등장 시

```python
[0.0, 0.0, 0.0]
```

초기화.

### select_action()

Epsilon-Greedy 정책.

### update_q()

Bellman Update 수행.

```python
q_values[action_idx] += alpha * (
    reward
    + gamma * max(next_q_values)
    - q_values[action_idx]
)
```

---

## Training Loop

학습 단계.

```text
reset

↓

state

↓

action

↓

step

↓

reward

↓

update_q

↓

next state
```

반복.

---

## Evaluation Loop

평가 단계.

차이점

```text
epsilon = 0

update_q 없음
```

즉

Q를 수정하지 않고

학습된 정책만 사용.

---

## 실험 결과

Q-Table이 실제로 학습됨.

예시

```python
Q[(0,1)]

=

[0.53, 0.89, 1.88]
```

Agent는

```text
action = 1
```

을 선택.

---

## 중요한 발견

done을 제거하고 계속 진행했을 때

Position이 크게 음수 방향으로 발산.

원인 분석:

학습은

```text
x ≈ 0 ~ 10
```

범위에서만 수행됨.

하지만 평가 시

```text
x = 20
x = 30
```

등 처음 보는 상태가 등장.

이 상태들은

```python
[0,0,0]
```

으로 초기화되어 있었음.

결과적으로

```text
학습 범위 밖 상태
=
처음 보는 상태
```

가 되었고 이상 행동 발생.

---

## Q-Table의 한계 발견

Q-Table은

```text
학습한 상태만 안다.
```

즉,

```text
Seen State
→ 가능

Unseen State
→ 불가능
```

---

## Project 09 결론

처음으로

```text
Reward

↓

Q Value

↓

Policy

↓

Action
```

이 연결되는 강화학습 시스템 구현 성공.

또한 Q-Table의 일반화 불가능 문제를 직접 관찰함.

이 문제를 해결하기 위해 다음 단계에서 Neural Network 기반 방법(DQN)이 등장한다.
