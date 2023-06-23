# ThemeIndex
테마 인덱스  
- 테마 정의
  - 종목이 보유한 키워드를 중심으로 군집화돼 거래가 발생한 이력이 있는 키워드
- 활용방안
  - Stock Factor와 결합 
  - 매크로 환경과의 상관관계 리서치 
  - 테마 트랜드 모니터링

# OutPutSample
### Data Table
**![image](https://github.com/song-junho/ThemeIndex/assets/67362481/35d4d13d-8043-40b2-8039-0ed87c3c1211)**

### Data Chart
![image](https://github.com/song-junho/ThemeIndex/assets/67362481/b1a511c7-9de9-4975-a763-c932ff3ce9b1)

# Developing Plan
### 1. Theme Clustering
 - 클러스터링 횟수가 증가해도 Lineaer 하게 평균거리가 감소하는 모습을 보인다. 
![image](https://github.com/song-junho/ThemeIndex/assets/67362481/b430d03d-6ec8-4513-bb41-7a5ad155c7dd)
 - 클러스터링 수를 100개로 지정한 예시  
![image](https://github.com/song-junho/ThemeIndex/assets/67362481/9227b7fb-9e35-428f-b929-c12f7e2ec646)

### 2. Macro 상관관계
 - 모델 학습 진행
   - x 변수: Macro pct_change(1m, 3m, 6m, 12m) or Macro 2year look back window z_score
   - y 변수: Theme 1M pct_change , (* shift 를 통해 +1달, +2달 등 조정)
     - 변화율의 범위로 분류 클래스 생성 ("1", "0", "-1")
   - 모델: SVC(SoftVectorClassifier) , RandomFroest , (?)
     
