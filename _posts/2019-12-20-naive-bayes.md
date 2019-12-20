---
layout: post
title: Naive Bayes
tags:
  - machine-learning
  - nlp
---
{% include katex.html %}

{% katexmm %}

Naive Bayes es un modelo de machine learning para clasificar textos entre una serie de categorías. Ejemplos de esto podrían ser libros de un autor, *tweets* de alguna persona, diálogos de personajes dentro de un texto, etc.

La bondad de este modelo es que a pesar de ser muy simplista, en la práctica puede arrojar resultados de gran calidad.

Para poder hacer uso del modelo, lo primero es contar con una base de datos con documentos, cada uno asignado a una categoría (o clase). Visto de otra forma, sea $d_n$ el $n$-ésimo documento de nuestra base y, $c_n$ la clase perteneciente al documento $d_n$, entonces nuestra base de datos estaría conformada por $\mathcal D = \{(d_n, c_n)\}_{n=1}^N$. La representación matemática de un documento $d_n$  está dado por un vector con las palabras únicas del documento, esto es $d=(w_1, \ldots, w_{|V|_n})$; donde $|V|_n$ denota la cardinalidad de $d_n$.

Dado un documento $d$ fuera de nuestra base de datos, queremos clasificarlo entre alguna de las $K$ clases. Una manera razonable de lograr eso sería eligiendo aquella clase con mayor probabilidad de la clase dado el documento. Esto es,

  $$
 \begin{aligned}
     \hat k &= \arg\max_k p(C_k|d)\\
     &= \arg\max_k  \frac{p(C_k)p(d|C_k)}{p(d)}\\
     &= \arg\max_k  p(C_k)p(d|C_k).
 \end{aligned} 
  $$

Es decir, la probabilidad de clasificar un nuevo documento $d$ está dado por el valor máximo que resulta en calcular la probabilidad *a priori* de una clase $k$ multiplicado por la verosimilitud del documento dada la clase $k$.

El siguiente paso sería definir quién es $p(C_k)$ y quién $p(d|C_k)$. Para $p(C_k)$ un valor *a priori* razonable sería considerar la proporción de textos que pertenecen a $C_k$. Sea $N_k$ el número de documentos pertenecientes a la clase $k$ y $N_D = \sum_k N_k$. Entonces, definimos la probabilidad *a priori* de un documento $k$ como

$$
    p(C_k) = \frac{N_k}{N_D}.
$$
Por otro lado, calcular $p(d|C_l) = p\left((w_1, \ldots, w_{|V|_n})|C_l\right)$ implica obtener la probabilidad del conjunto de palabras. Es en este momento que el modelo de *Naïve Bayes* toma forma. A fin de obtener una verosimilitud para $d$, haremos dos grandes suposiciones sobre las propiedades de $p(d|C_k)$

1. **La suposición de la bolsa de palabras**: En primera instancia, asumiremos que el orden de las palabras, condicionado a una clase, no son de importancia, es decir, para dos palabras $w_i$, $w_j$, $p(w_i, w_j|C_k) = p(w_j, w_i|C_k)$

2. **La suposición de *Naive Bayes***:  Condicionado a una clase $k$, cada una de las palabras son independientes. Es decir, dada dos palabras $w_i$, $w_j$, $p(w_i, w_j|C_k) = p(w_j|C_k) p(w_i|C_k)$

Con estas dos suposiciones podemos decir que dado un nuevo documento $d$ , lo clasificamos como la clase

$$
\begin{aligned}
\hat k &=   \arg\max_k p(C_k) \prod_{v \in V_d} p(w_v|C_k)\\
            &= \arg\max_k \log p(C_k) + \sum_{v\in V_d } \log p(w_v|C_k)
\end{aligned}
$$

donde $V_d$ es el vocabulario del documento $d$.

-----

## Estimando $p(w_v|C_k)$
La probabilidad $p(w_v|C_k)$ la conseguiremos a partir de máxima verosimilitud. Considera

$$
\begin{aligned}
    p(w_v|C_k) &= \frac{p(w_v, C_k)}{p(C_k)}\\
                      &= \frac{p(w_v^k)}{p(C_k)}\\
                      &= \frac{p(w_v^k)}{\sum_{v\in V_k}p(w_v^k)}
\end{aligned}
$$

donde denotamos $w_v^k = (w_v, C_k)$.

Asumamos $w_v^k \sim    \text{Bern}(\mu_v^k)$.  Luego, la máxima verosimilitud de $\mu_v^k$ estaría dada por

$$
\begin{aligned}
    \hat \mu_v^k &= \arg\max_{\mu_v^k } p(D|\mu_v^k) \\
                &= \arg\max_{\mu_v^k } \prod_n p(d_n|\mu_v^k) \\
                &= \arg\max_{\mu_v^k } \prod_{w_m\in W} p(w_m|\mu_v^k) \\
\end{aligned}
$$

Notemos que la primera y segunda igualdad corren sobre cada documento y la última igualdad corre sobre cada palabra, dentro de cada documento.

Antes de proseguir, denotemos $\bold{1}(n)_{v,k} = \bold{1}_{(w_n = w_v) \vee (w_n \in V_k)}$, la indicadora de una palabra $w_m$ dentro de la base de datos que sea igual a $w_v$ y que pertenezca al vocabulario de elementos de la clase $k$. Entonces,
$$
\begin{aligned}
                \hat \mu_v^k &= \arg\max_{\mu_v^k } \sum_m \log p(w_m|\mu_v^k) \\
                &= \arg\max_{\mu_v^k } \sum_m  \bold{1}(n)_{v,k} \log \mu_v^k + (1 -  \bold{1}(n)_{v,k})\log(1 - \mu_v^k)\\
                &= \arg\max_{\mu_v^k } C(w_v^k) \log \mu_v^k + (1 -  C(w_v^k))\log(1 - \mu_v^k)\\
                &=\frac{C(w_v^k)}{N}
\end{aligned}
$$

Dónde $C(w_v^k)$ es un contador del número de palabras de $w_v$ dentro de documentos de la clase $k$.

Con este último resultado podemos ver que
$$
\begin{aligned}
    p(w_v|C_k) &= \frac{C(w_v^k)/N}{\sum_{\hat v\in V}C(w_{\hat v}^k)/N}\\
    &= \frac{C(w_v^k)}{\sum_{\hat v\in V_k}C(w_{\hat v}^k)}
\end{aligned}
$$

Es decir, la probabilidad de una palabra $w_v$ dado un documento de la clase $k$ está dado por cuántas veces aparece la palabra $w_v$ dentro de la clase $k$, divido por el total de palabras que se dijeron dentro de la clase $k$.

Parecería entonces que ya tenemos un modelo sobre el cuál podemos estimar una clase dado un documento $d$, pero, ¿qué pasa si el documento $d$ tiene información relevante sobre una clase $\hat k$ y exista una palabra $\hat w_j$ que no existe dentro de $V_k$? En este caso, $p(\hat w_j|V_k)$ sería cero y, subsecuentemente, $p(C_k|d) =0$. Para evitar esta clase de problemas, podemos *suavizar* las probabilidades de por el método de la suavización de Laplace (Laplace Smoothing)

-----

## La suavización de Laplace
Consideremos nuevamente $w_v^k\sim  \text{Bern}(\mu_v^k)$. Asumiremos *a priori* $\mu_v^k\sim U(0,1)$. Entonces, queremos calcular la probabilidad a posteriori de $\mu_v^k$. Esto es

$$
\begin{aligned}
    p(\mu_v^k|\mathcal{D}) &\propto p(\mathcal{D}|\mu_v^k) p(\mu_v^k)\\
        &=(\mu_v^k)^{C(w_v^k)} (1 - \mu_v^k)^{N - C(w_v^k)}
\end{aligned}
$$

Con esto último podemos ver que 
$$
\mu_v^k|\mathcal{D}\sim\text{Beta}(C(w_v^k) + 1, N - C(w_v^k) + 1).
$$
Cuya esperanza está dada por
$$
    \mathbb{E}[\mu_v^k|\mathcal{D}] =\frac{C(w_v^k) + 1}{N + 2}.
$$

Este último resultado sería una manera alternativa a elegir $\mu_v^k$ dado por máxima verosimilitud. De esta forma

$$
        p(w_n|C_k) = \frac{C(w_v^k) + 1}{\sum_{\hat v\in V_k}C(w_{\hat v}^k) + |V|}
$$

garantiza una probabilidad (aunque muy pequeña) para cada palabra que veamos el algún nuevo documento.

Finalmente, nuestro método para determinar la clase $k$ de un documento estaría dado por:
$$
\hat k= \arg\max_k \log \frac{N_k}{N_D} + \sum_{v\in V_d } \log \frac{C(w_v^k) + 1}{\sum_{\hat v\in V_k}C(w_{\hat v}^k) + |V|}
$$


{% endkatexmm %}
