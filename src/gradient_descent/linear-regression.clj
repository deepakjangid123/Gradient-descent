(ns gradient-descent.linear-regression
  (:require
    [clojure.core.matrix :as matrix]
    [clojure.data.csv :as csv]
    [clojure.java.io :as io]
    [incanter.charts :as charts]
    [incanter.core :refer [view]]))

(defn read-csv
  "Reads csv file and returns its data"
  []
  (with-open [in-file (io/reader "resources/linear-regression.csv")]
    (doall
      (csv/read-csv in-file))))

;; Parameters needed for linear-regression
(def data (read-csv))
(def alpha 0.01)
(def iterations 1500)
(def xs (map (comp read-string first) data))
(def y (map (comp read-string second) data))
(def number-of-examples (count xs))

;; Now create X matrix
(def X (-> [(repeat number-of-examples 1) xs]
           (matrix/matrix)
           (matrix/transpose)))

;; Thetha values
(def initial-theta [0 0])

;; Now hypothesis is multiplication of theta and X
(defn hypothesis
  "Returns hypothesis for given `thetha` and `X`"
  [theta X]
  (matrix/mmul theta (matrix/transpose X)))

(defn mean-square-error
  "Returns mean-square-error for given `guess` matrix(h(x)) and
  `actual` matrix(y)"
  [guess actual]
  (-> (matrix/sub guess actual)
      (matrix/square)
      (matrix/esum)))

(defn compute-cost
  "Computes cost function J(theta)"
  [X y theta]
  (let [predicted (hypothesis theta X)]
    (/ (mean-square-error predicted y)
       (* 2 number-of-examples))))

(defn cost-derivative
  "Compute cost J(theta)'s derivative"
  [X y theta]
  (let [prediction (hypothesis theta X)]
    (-> prediction
        (matrix/sub y)
        (matrix/mmul X)
        (matrix/mul (/ alpha number-of-examples)))))

(defn gradient-descent
  "Returns the result for gradient-descent algorithm"
  [X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta]
    (if (zero? remaining-iterations)
      theta
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))))))

(defn costs
  "Keeps track of cost fn value at each step"
  [X y theta alpha number-of-iterations]
  (loop [remaining-iterations number-of-iterations
         theta theta
         costs []]
    (if (zero? remaining-iterations)
      costs
      (recur (dec remaining-iterations)
             (matrix/sub theta (cost-derivative X y theta))
             (conj costs (compute-cost X y theta))))))

(def result (gradient-descent X y initial-theta alpha iterations))
(def cost-history (costs X y initial-theta alpha iterations))

(defn plot-cost-history
  "Plots cost-history at each and every step"
  [y]
  (let [plot (charts/scatter-plot (range 0 iterations) y
                                  :x-label "Iteration"
                                  :y-label "Cost")]
    (doto plot view)))

(plot-cost-history cost-history)
(defn plot-data
  "Plots the data-points"
  [x y]
  (let [plot (charts/scatter-plot x y
                                  :x-label "Population of the city in 10,000's"
                                  :y-label "Profit in $10,000s")]
    (doto plot
      (charts/add-function #(+ (first result) (* (second result) %)) 0 25)
      view)))

(plot-data xs y)
