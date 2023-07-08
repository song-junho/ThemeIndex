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
   
- 모델 학습 결과(Randmforest)
   - 테마별 train, test_score  
       ![image](https://github.com/song-junho/ThemeIndex/assets/67362481/175fabec-3fb4-4ea2-9ad8-922b11fac477)
     
- 모델 학습 결과(XGBoost)
   - 테마별 train, test_score  
       
        | key_nm             |   train_score |   test_score |
        |:-------------------|--------------:|-------------:|
        | 도시가스           |             1 |     0.94     |
        | 술                 |             1 |     0.94     |
        | 소주               |             1 |     0.94     |
        | 음료               |             1 |     0.94     |
        | 온라인             |             1 |     0.92     |
        | 방직               |             1 |     0.92     |
        | 음식료             |             1 |     0.92     |
        | 포장               |             1 |     0.9      |
        | 제지               |             1 |     0.9      |
        | 금리               |             1 |     0.9      |
        | 쇼핑몰             |             1 |     0.9      |
        | 가스               |             1 |     0.9      |
        | 라면               |             1 |     0.9      |
        | 교육               |             1 |     0.9      |
        
    - ex) 7월에 강할 예상 테마
    
      | theme_nm           |   score |   model_score |   pred_proba |
      |:-------------------|--------:|--------------:|-------------:|
      | 통신               |       2 |      0.86     |     0.91197  |
      | 3D프린터           |       2 |      0.82     |     0.944172 |
      | LPG                |       2 |      0.8      |     0.905898 |
      | 철강원재료         |       2 |      0.8      |     0.916655 |
      | 식자재             |       2 |      0.78     |     0.932695 |
      | 언택트             |       2 |      0.76     |     0.909345 |
      | 신재생             |       2 |      0.74     |     0.921915 |
      | 무역               |       2 |      0.72     |     0.953188 |
      | 건설기계           |       2 |      0.72     |     0.906222 |
      
    - ex) LPG 의 key macro factor
    
        ![image](https://github.com/song-junho/ThemeIndex/assets/67362481/e22db9d5-9f04-4de9-8fc8-ec039a5ed51f)
        