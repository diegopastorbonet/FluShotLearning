# FluShotLearning
Predicción de vacunación contra la gripe mediante machine learning

En este proyecto, buscamos predecir si una persona se vacunará contra la gripe H1N1 y la gripe estacional basándonos en datos demográficos y de comportamiento. Esta predicción es crucial para identificar grupos de alto riesgo y dirigir las campañas de vacunación de manera más efectiva.

El análisis se basa en el conjunto de datos "Flu Shot Learning" que contiene información sobre varias características relevantes, incluyendo la preocupación por el H1N1, el conocimiento sobre la gripe, y el acceso a servicios médicos.

Objetivos
Analizar los factores que influyen en la vacunación contra el H1N1 y la gripe estacional.
Desarrollar modelos predictivos para ambas vacunas.
Evaluar el rendimiento de los modelos y proponer mejoras.
Datos obtenidos del sitio web de Driven Data.

https://www.drivendata.org/competitions/66/flu-shot-learning/data/

**Autor**: Diego Pastor Bonet

**Datos del modelo final elegido**
* Score test (DrivenData): 0.7860
* Algoritmo ML: Regresión Lineal, RandomSearch
* Hiperparámetros:
  * ridge alpha = 0.1
  * ridge fit intercept = True

Lista de características:

* income_poverty_num             
* education_num                  
* employment_status_num          
* h1n1_knowledge                 
* household_adults               
* children_per_adult             
* household_size                 
* health_worker                  
* sex_bin                                       
* census_msa_num                 
* opinion_h1n1_vacc_effective                      
* household_children             
* opinion_seas_vacc_effective    
* behavioral_avoidance           
* doctor_recc_h1n1               
* behavioral_wash_hands          
* doctor_recc_seasonal          
* respondent_id                 
* child_under_6_months          
* opinion_seas_risk             
* opinion_h1n1_risk             
* hhs_geo_region_num            
* age_group_num                 
* behavioral_touch_face         
* h1n1_concern                  
* behavioral_face_mask          
* behavioral_antiviral_meds     
* opinion_h1n1_sick_from_vacc   
* chronic_med_condition         
* opinion_seas_sick_from_vacc   
* precaution_level              
* race_num                      
* behavioral_large_gatherings   
* behavioral_outside_home       
* employment_occupation_num     
* employment_industry_num      
* rent_or_own_bin               
* marital_status_bin   



 Resumen dataset (transformaciones, etc.):
  * Binarización de las variables 'sex', 'marital_status' y 'rent_or_own' dando lugar a 'sex_bin', 'marital_status_bin' y 'rent_or_own_bin'.
  * Transformación a escala numérica las variables income_poverty_num,        education_num, employment_status_num, employment_industry_num, census_msa_num, employment_occupation_num, hhs_geo_region_num y age_group_num.
  * Tratamiento de nulos con el método ffill, excepto en el caso de las variables 'income_poverty_num', 'doctor_recc_h1n1' y 'doctor_recc_seasonal' en las que se han hecho interpolaciones.
  * Creación de las variables 'precaution_level' y 'household_size'
  * Eliminación de la variable 'health_insurance'.

---
# Resumen
---

A lo largo de este análisis se elaborará un modelo de predicción que podrá ser puesto a prueba en la web de DrivenData.

En primer lugar se hará un procesado teniendo en cuenta las necesidades estructurales de la competición.

Se probarán algoritmos de agrupamiento para generar etiquetas adicionales en el dataset, se probarán también varios modelos de predicción para encontrar el mejor posible. Finalmente, todo esto será puesto a prueba mediante un conjunto de validación y adicionalmente por la web mediante un conjunto de test. La métrica que se utilizará para evaluar los modelos será el Score ([0, 1]) aportado por DrivenData.

El objetivo será alcanzar el mayor valor de Score entregando diferentes intentos con combinaciones de grupos de características y algoritmos a esta web.

En base a esta métrica, el conjunto de datos mejor valorado ha sido el conformado por el conjunto de datos original, al que se le ha sin etiquetas adicionales generadas por algoritmos de agrupamiento, al que se han añadido unas nuevas características en base a las previas. Otras han sido eliminadas por diferentes motivos como excesiva proporción de nulos. El modelo de predicción que mejor ha funcionado es el de Regresión Lineal optimizado mediante Random Search.

Otras combinaciones como una sub-selección de características, diferentes modelos de predicción o diferentes algoritmos de agrupamiento han dado todos peores resultados que éste, sin embargo, aún hay espacio para seguir investigando, especialmente con diferentes conjuntos de características.

---
#Resultados
---

Se muestra una tabla de comparación de resultados basándonos en la evaluación del sitio web de driven data

| Modelo | Score (Driven Data) | Algoritmo | Hiperparámetros** |
| --- | --- | --- | --- 
| multi_output_svm | 0.5027 | SVM | estimator = SVR, n_jobs = None, kernel = rbf, degree = 3, gamma = scale, coef0 = 0.0, C = 1.0, epsilon = 0.1, shrinking = True, tol = 1e-3, max_iter = -1
| knn_model | 0.5000 | KNN | n_neighbors=**5**
| rf_model | 0.7516 | Random Forest | n_estimators = 100, random_state = 42
| linear_model | 0.7859 | Regresión Lineal | fit_intercept = True, copy_X = True, n_jobs = None
| best_model_grid | 0.7860 | Grid Search - Regresión Lineal | alpha = 0.1, intercept = True
| best_model_random  | 0.7859 | Random Search - Regresión Lineal | alpha = 0.005537420375935475, fit_intercept = True
| ridge_model | 0.7859 | Optimización Bayesiana - Regresión Lineal | alpha = 0.00010034426392908844, fit_intercept = True

** Los hiperparámetros de los modelos multi_output_svm y linear_model son los que vienen por defecto con la biblioteca sklearn.

![image](https://github.com/user-attachments/assets/68557937-3154-4366-9674-d77bf92d5e2e)

A continuación se muestra una nueva tabla con los resultados de cada modelo respecto del conjunto de validación.

| Modelo             | F1 Score H1N1 | Precisión H1N1 | Recall H1N1 | AUC H1N1 | F1 Score Seasonal | Precisión Seasonal | Recall Seasonal | AUC Seasonal |
| ------------------ | -------------- | -------------- | ----------- | -------- | ----------------- | ------------------ | ---------------- | ------------ |
| multi_output_svm   | 0.0000         | 0.0000         | 0.0000      | 0.53     | 0.0000            | 0.0000             | 0.0000           | 0.72         |
| knn_model          | 0.1136         | 0.2557         | 0.0730      | 0.54     | 0.5047            | 0.5172             | 0.4929           | 0.55         |
| rf_model           | 0.5256         | 0.6486         | 0.4418      | 0.82     | 0.7444            | 0.7537             | 0.7353           | 0.83         |
| linear_model       | 0.5056         | 0.6947         | 0.3974      | 0.82     | 0.7429            | 0.7680             | 0.7194           | 0.84         |
| best_model_grid    | 0.5056         | 0.6947         | 0.3974      | 0.82     | 0.7429            | 0.7680             | 0.7194           | 0.84         |
| best_model_random  | 0.5056         | 0.6947         | 0.3974      | 0.82     | 0.7429            | 0.7680             | 0.7194           | 0.84         |
| ridge_model        | 0.5056         | 0.6947         | 0.3974      | 0.82     | 0.7429            | 0.7680             | 0.7194           | 0.84         |

![image](https://github.com/user-attachments/assets/ab2bfdec-fc6b-4f17-8430-fa600dc79b1e)

![image](https://github.com/user-attachments/assets/9d0cb01f-e602-441e-846b-732ec17f3b52)

Los modelos ridge_model y rf_model son superiores en cuanto a la evaluación del conjunto de validación, esto contrasta con la evaluación de la web DrivenData, que da un Score superior al modelo best_model_grid.

---
#Resumen final y conclusiones
---

Las características finales de los modelos son los siguientes:

* income_poverty_num             
* education_num                  
* employment_status_num          
* h1n1_knowledge                 
* household_adults               
* children_per_adult             
* household_size                 
* health_worker                  
* sex_bin                                      
* census_msa_num                 
* opinion_h1n1_vacc_effective                      
* household_children             
* opinion_seas_vacc_effective    
* behavioral_avoidance           
* doctor_recc_h1n1               
* behavioral_wash_hands          
* doctor_recc_seasonal          
* respondent_id                 
* child_under_6_months          
* opinion_seas_risk             
* opinion_h1n1_risk             
* hhs_geo_region_num            
* age_group_num                 
* behavioral_touch_face         
* h1n1_concern                  
* behavioral_face_mask          
* behavioral_antiviral_meds     
* opinion_h1n1_sick_from_vacc   
* chronic_med_condition         
* opinion_seas_sick_from_vacc   
* precaution_level              
* race_num                      
* behavioral_large_gatherings   
* behavioral_outside_home       
* employment_occupation_num     
* employment_industry_num      
* rent_or_own_bin               
* marital_status_bin      

El mejor modelo es por tanto best_model_grid puesto que obtiene el mejor resultdo en la validación de Score de la página web de DrivenData.

Al poder contar con varios algoritmos de aprendizaje supervisado, como lo son SVM, KNN, el Random Forest o la Regrsión Lineal hemos sido capaces de identificar cuál genera mejor resultado y proceder a optimizarlo.

Lo mismo ha sucedido con los algoritmos de agrupamiento que utilizamos en la actividad anterior. Se ha comprobado que K Means y DBScan no aportaban información útil mediante los agrupamientos.

Cabe señalar que la evaluaciones de Score en el conjunto de validación no se corresponde con la de la web, sino que es en general superior, esto es curioso dado que los datos del conjunto de validación no alimentan los modelos y no debería de haber overfitting.

Sin embargo, a la vista de los resultados, queda claro que se ha incurrido en alguna clase de sesgo durante la elaboración de los modelos, tal vez algún overfitting que no somos capaces de detectar.

Cabe también mencionar que la optimización de hiperparámetros mediante varias técnicas no afecta nada en absoluto a los resultados de Score. Esto puede ser porque existe una tendencia muy marcada a alcanzar ese resultado que no puede ser modificada por estos parámetros.

El resultado final de Score = 0.7860 de DrivenData se considera exitoso pero con amplio margend e mejora. Cabe resaltar el cambio de 0.5000 obtenido con el primer modelo sin optimizar respecto del mejor modelo, ya trabajado y optimizado.

Es posible que se pudiera haber mejorado aún más mediante nuevas selecciones de características distintas a la realizada, sin embargo, la limitación de intentos en la página web ha sido un escollo en este aspecto, puesto que hay una limitación de tres intentos al día.

Como conclusión general, hemos desarrollado modelos predictivos para la vacunación contra el H1N1 y la gripe estacional. Aunque los modelos lograron un rendimiento razonable, hay espacio para mejoras. Por ejemplo, se podría explorar la recolección de datos adicionales o la inclusión de características más avanzadas que capturen mejor la variabilidad de los datos.

---
#Trabajo Futuro
---

Se podría ahondar mucho más en la selección de características ahora que se tiene un buen algoritmo de regresión optimizado.

Con una cantidad suficiente de pruebas y ensayos con diferentes colecciones de características se alcanzaría un Score superior.

Se deja como trabajo futuro:
1. Explorar modelos más avanzados como XGBoost o redes neuronales.
2. Realizar una búsqueda de hiperparámetros más exhaustiva utilizando GridSearchCV.
3. Evaluar la posibilidad de combinar varios modelos en un ensamble para mejorar el rendimiento.
4. Explorar selección de características.



