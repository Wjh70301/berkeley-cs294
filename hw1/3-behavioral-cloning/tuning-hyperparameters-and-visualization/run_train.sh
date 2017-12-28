# layer number
for i in {1..10..1}; do
    # layer size
    for j in {1..100..2}; do
        echo "python train_humanoid.py ${i} ${j} ../train_test_data/humanoid_train_test.pkl"
    done
done
