
# PCA

## 주성분분석(PCA)

PCA는 데이터의 분산(variance)을 최대한 보존하면서 서로 직교하는 새 기저(축)를 찾아, 고차원 공간의 표본들을 선형 연관성이 없는 저차원 공간으로 변환하는 기법입니다.

<img src="images/PCA.png" width="50%" style="display: block; margin: auto;" />

그림에서 알 수 있듯이, 2차원 공간에 있는 데이터들이 하나의 주성분(PC1)을 새로운 기저로 선형변환된 걸 확인할 수 있습니다. 여기에서 1시 방향의 사선축이 원 데이터의 분산을 최대한 보존하는 새로운 기저입니다. 두 번째 성분은 첫 번째 성분의 방향에 직각인 방향으로 분산을 최대화하는 선형 결합을 선택하는 식으로 만들며, 그 후 성분들도 변수 수만큼 같은 방법으로 만듭니다. PCA의 목적은 바로 이런 축을 찾는 데 있습니다.

## iris 데이터 분석

먼저 R의 기본 데이터인 iris 데이터를 통해 PCA 분석을 해보도록 합니다.

### 데이터 불러오기


```r
library(magrittr)
library(corrplot)

data("iris")
iris.scale = iris[, 1:4] %>%
  scale()

iris.scale %>%
  cor() %>%
  corrplot.mixed()
```

<img src="07-PCA_files/figure-html/unnamed-chunk-3-1.png" width="50%" style="display: block; margin: auto;" />

먼저 데이터를 평균 0, 표준편차 1로 표준화한 후, 상관관계를 구합니다. Petal.Width와 Petal.Length 간, Sepal.Length와 Petal.Length 간에 높은 상관관계가 있습니다.

### 모형화

PCA는 다음 단계를 거친다.

1. 성분 추출 및 남길 성분의 수 결정
2. 남은 성분을 회전
3. 회전된 결과를 해석
4. 요인 점수를 생성
5. 요인 점수를 입력 변수로 사용해 회귀 분석을 하고, 테스트 데이터에 관한 평가

R에서는 기본함수인 `prcomp()`, `princomp()` 함수를 통해 PCA 분석을 수행할 수 있으며, `psych` 패키지를 통해 더욱 다양한 분석을 할 수 있습니다. 먼저 기본함수인 `prcomp()` 함수를 이용해 성분을 추출합니다.


```r
pca.iris = prcomp(iris.scale)
summary(pca.iris)
```

```
## Importance of components:
##                         PC1   PC2    PC3     PC4
## Standard deviation     1.71 0.956 0.3831 0.14393
## Proportion of Variance 0.73 0.229 0.0367 0.00518
## Cumulative Proportion  0.73 0.958 0.9948 1.00000
```

Cumulative Proportion을 살펴보면, PC1 만으로도 73% 가량의 분산을 설명하며, PC 2까지는 95%의 분산을 설명합니다.

몇번째 성분까지 사용하는 것이 좋은지를 판단하기 위해 screeplot을 그려주도록 합니다.


```r
screeplot(pca.iris, type = "l")
```

<img src="07-PCA_files/figure-html/unnamed-chunk-5-1.png" width="50%" style="display: block; margin: auto;" />

흔히 기울기가 달라지며 꺽이는 지점을 Elbow Point라고 부르는데 보통 이 부분의 PC까지를 사용해서 변수를 축소합니다. 2개 성분만으로도 충분하며, 3개 성분으로 대부분을 설명할 수 있습니다.

이번에는 `biplot()` 함수를 이용해 행렬도를 그려보도록 합니다.


```r
biplot(pca.iris)
```

<img src="07-PCA_files/figure-html/unnamed-chunk-6-1.png" width="50%" style="display: block; margin: auto;" />

위 그림은 각 개체에 대한 첫번째, 두번째 성분에 대한 점수 및 행렬도를 나타낸 것으로써, 가까운 거리와 방향일수록 변수들의 상관성이 높습니다. 

마지막으로 각 성분에 대한 기여도를 출력합니다.


```r
pca.iris$rotation
```

```
##                  PC1      PC2     PC3     PC4
## Sepal.Length  0.5211 -0.37742  0.7196  0.2613
## Sepal.Width  -0.2693 -0.92330 -0.2444 -0.1235
## Petal.Length  0.5804 -0.02449 -0.1421 -0.8014
## Petal.Width   0.5649 -0.06694 -0.6343  0.5236
```

## 북미 프로 아이스하키 리그 데이터 분석

북미 프로 아이스하키 리그에 관한 데이터를 분석합니다.

### 데이터 불러오기

해당 데이터는  https://github.com/datameister66/data 에서 nhlTrain.csv와 nhlTest.csv 데이터를 다운받도록 합니다.


```r
train = read.csv('https://raw.githubusercontent.com/datameister66/data/master/NHLtrain.csv')
test = read.csv('https://raw.githubusercontent.com/datameister66/data/master/NHLtest.csv')
```

각 피처는 다음과 같습니다.

- Team: 팀의 연고지
- ppg: 게임당 점수의 평균
- Goals_For: 팀의 경기당 평균 득점
- Goals_Against: 팀의 경기당 평균 실점
- Shots_For: 경기당 팀의 골 근처에서 슛을 한 횟수
- Shots_Against: 경기당 팀이 골 근처에서 상대팀의 슛을 허용한 횟수
- PP_perc: 파워플레이 상황에서 팀이 득점한 퍼센트
- PK_perc: 상대팀이 파워플레이 상황일 때 실점을 하지 않은 시간의 퍼센트
- CF60_pp: 팀의 파워플레이 60분당 Corsi 점수. Corsi 점수는 Shots_For와 상대에게 막힌 것이나 네트를 벗어난 것의 개수를 합한 것이다.
- CA60_sh: 상대 팀의 파워플레이 60분당 Corsi 점수
- OZFOperc_pp: 팀이 파워플레이 상황일 때 공격자 지역에서 시합이 재개된 퍼센트
- Give: 팀이 경기당 퍽을 준 평균 횟수
- Take: 팀이 경기당 퍽을 가져온 평균 횟수
- hits: 팀의 경기당 보디체크 평균 횟수
- blks: 팀의 경기당 상대방 슛을 블로키한 횟수의 평균

데이터를 평균 0, 표준편차 1로 표준화한 후, 상관관계를 구하도록 합니다.


```r
library(magrittr)

train.scale = scale(train[, -1:-2])
nhl.cor = cor(train.scale)
nhl.cor %>% corrplot()
```

<img src="07-PCA_files/figure-html/unnamed-chunk-9-1.png" width="50%" style="display: block; margin: auto;" />

### 성분 추출


```r
library(psych)

pca = principal(train.scale, rotate = 'none')
plot(pca$values, type = 'b', ylab = 'Eigenvalues', xlab = 'Component')
```

<img src="07-PCA_files/figure-html/unnamed-chunk-10-1.png" width="50%" style="display: block; margin: auto;" />

`psych` 패키지의 `principal()` 함수를 통해 성분을 추출하도록 합니다. 5개 성분만으로도 충분한 설명력이 있는 것으로 보입니다.

### 직각 회전과 해석

회전의 목적은 특정한 성분에 관해 변수의 기여도를 최대화함으로써 각 성분 사이의 상관 관계를 줄여 해석을 간단학 하는 것입니다.


```r
pca.rotate = principal(train.scale, nfactors = 5, rotate = 'varimax')
print(pca.rotate)
```

```
## Principal Components Analysis
## Call: principal(r = train.scale, nfactors = 5, rotate = "varimax")
## Standardized loadings (pattern matrix) based upon correlation matrix
##                 RC1   RC2   RC5   RC3   RC4   h2   u2 com
## Goals_For     -0.21  0.82  0.21  0.05 -0.11 0.78 0.22 1.3
## Goals_Against  0.88 -0.02 -0.05  0.21  0.00 0.82 0.18 1.1
## Shots_For     -0.22  0.43  0.76 -0.02 -0.10 0.81 0.19 1.8
## Shots_Against  0.73 -0.02 -0.20 -0.29  0.20 0.70 0.30 1.7
## PP_perc       -0.73  0.46 -0.04 -0.15  0.04 0.77 0.23 1.8
## PK_perc       -0.73 -0.21  0.22 -0.03  0.10 0.64 0.36 1.4
## CF60_pp       -0.20  0.12  0.71  0.24  0.29 0.69 0.31 1.9
## CA60_sh        0.35  0.66 -0.25 -0.48 -0.03 0.85 0.15 2.8
## OZFOperc_pp   -0.02 -0.18  0.70 -0.01  0.11 0.53 0.47 1.2
## Give          -0.02  0.58  0.17  0.52  0.10 0.65 0.35 2.2
## Take           0.16  0.02  0.01  0.90 -0.05 0.83 0.17 1.1
## hits          -0.02 -0.01  0.27 -0.06  0.87 0.83 0.17 1.2
## blks           0.19  0.63 -0.18  0.14  0.47 0.70 0.30 2.4
## 
##                        RC1  RC2  RC5  RC3  RC4
## SS loadings           2.69 2.33 1.89 1.55 1.16
## Proportion Var        0.21 0.18 0.15 0.12 0.09
## Cumulative Var        0.21 0.39 0.53 0.65 0.74
## Proportion Explained  0.28 0.24 0.20 0.16 0.12
## Cumulative Proportion 0.28 0.52 0.72 0.88 1.00
## 
## Mean item complexity =  1.7
## Test of the hypothesis that 5 components are sufficient.
## 
## The root mean square of the residuals (RMSR) is  0.08 
##  with the empirical chi square  28.59  with prob <  0.19 
## 
## Fit based upon off diagonal values = 0.91
```

먼저 각 5개 성분에 관한 변수들의 기여도는 RC1부터 RC5까지 열로 구분되어 있습니다. RC1(성분1)은 **Goals_Against**와 **Shots_Against** 변수의 성분에 관한 기여도가 높은 양의 값이고, **PP_perc**와 **PK_perc** 변수의 기여도가 높은 음의 값입니다.

RC2(성분2)은 **Goals_For** 변수가 높은 기여도를 가지고 있으며, RC5는 **Shots_For, CF60_pp, OZFOperc_pp**가 높은 기여도를 갖고 있습니다. RC3는 **take** 변수만 연관이 있으며, RC4는 **hits** 변수와 연관이 있습니다.

SS loadings 제곱합으로 시작하는 표의 숫자는 각 성분의 고윳값(Eigenvalue) 입니다. 이 고윳값이 정규화되면 Proportion Explained 행의 값이며, 성분 1이 5개의 회전된 성분 모두가 설명하는 분산의 28%를 설명하는 것을 볼 수 있습니다.

경험적으로 선택된 성분들이 설명하는 분산의 총합이 최소한 전체 분산의 70%를 넘어야 하며, Cumulative Var 행의 5개 성분 합이 총 74%의 분산을 나타내고 있습니다.

### 요인 점수 생성

회전된 성분들의 기여도를 각 팀의 요인 점수로 변환합니다.


```r
pca.scores = data.frame(pca.rotate$scores)
head(pca.scores)
```

```
##       RC1       RC2     RC5     RC3      RC4
## 1 -2.2153  0.002821  0.3162 -0.1572  1.52780
## 2  0.8815 -0.569239 -1.2361 -0.2703 -0.01132
## 3  0.1032  0.481754  1.8135 -0.1607  0.73465
## 4 -0.0663 -0.630676 -0.2121 -1.3086  0.15413
## 5  1.4966  1.156906 -0.3222  0.9647 -0.65648
## 6 -0.4890 -2.119952  1.0456  2.7375 -1.37358
```

독립변수(png)를 데이터의 열로 불러오도록 합니다.


```r
pca.scores$ppg = train$ppg
```

### 회귀 분석


```r
nhl.lm = lm(ppg ~ ., data = pca.scores)
summary(nhl.lm)
```

```
## 
## Call:
## lm(formula = ppg ~ ., data = pca.scores)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.16327 -0.04819  0.00372  0.03872  0.16591 
## 
## Coefficients:
##             Estimate Std. Error t value   Pr(>|t|)    
## (Intercept)  1.11133    0.01575   70.55    < 2e-16 ***
## RC1         -0.11220    0.01602   -7.00 0.00000031 ***
## RC2          0.07099    0.01602    4.43    0.00018 ***
## RC5          0.02295    0.01602    1.43    0.16500    
## RC3         -0.01778    0.01602   -1.11    0.27804    
## RC4         -0.00531    0.01602   -0.33    0.74300    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.0863 on 24 degrees of freedom
## Multiple R-squared:  0.75,	Adjusted R-squared:  0.698 
## F-statistic: 14.4 on 5 and 24 DF,  p-value: 0.00000145
```

$R^2$가 70%에 달하며, p값이 1.446e-06로 나와 통계적으로 높은 유의성을 갖고 있습니다. 그러나 RC1과 RC2를 제외한 3개 성분은 유의하지 않은 것으로 보입니다. 두 가지 피처만을 대상으로 회귀분석을 다시 실시합니다.


```r
nhl.lm2 = lm(ppg ~ RC1 + RC2, data = pca.scores)
summary(nhl.lm2)
```

```
## 
## Call:
## lm(formula = ppg ~ RC1 + RC2, data = pca.scores)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -0.1891 -0.0443  0.0144  0.0565  0.1647 
## 
## Coefficients:
##             Estimate Std. Error t value   Pr(>|t|)    
## (Intercept)   1.1113     0.0159   70.04    < 2e-16 ***
## RC1          -0.1122     0.0161   -6.95 0.00000018 ***
## RC2           0.0710     0.0161    4.40    0.00015 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.0869 on 27 degrees of freedom
## Multiple R-squared:  0.715,	Adjusted R-squared:  0.694 
## F-statistic: 33.8 on 2 and 27 DF,  p-value: 0.000000044
```

$R^2$가 역시나 70%에 가까우며, 통계적으로 유의한 모델입니다.


```r
plot(nhl.lm2$fitted.values, train$ppg,
     xlab = 'Predicted',
     ylab = 'Actual')
```

<img src="07-PCA_files/figure-html/unnamed-chunk-16-1.png" width="50%" style="display: block; margin: auto;" />


```r
sqrt(mean(nhl.lm2$residuals^2))
```

```
## [1] 0.08244
```

평균 제곱 오차의 제곱근을 계산한 후 테스트 셋과 비교해보도록 합니다.


```r
test.scores = data.frame(predict(pca.rotate, test[, c(-1:-2)]))

test.scores$pred = predict(nhl.lm2, test.scores)
test.scores$ppg = test$ppg
```


```r
plot(test.scores$pred, test.scores$ppg,
     xlab = 'Predicted',
     ylab = 'Actual')
```

<img src="07-PCA_files/figure-html/unnamed-chunk-19-1.png" width="50%" style="display: block; margin: auto;" />


```r
resid = test.scores$ppg - test.scores$pred
sqrt(mean(resid^2))
```

```
## [1] 0.1012
```
