"""View Federated Learning Results"""
import json

with open('federated_learning_results.json') as f:
    results = json.load(f)

print('\n' + '='*70)
print('FEDERATED LEARNING SYSTEM - RESULTS')
print('='*70)

config = results['system_config']
print(f'\nSystem Configuration:')
print(f'  Clients trained:       {config["num_clients"]}')
print(f'  Total samples:         {config["total_samples"]:,}')
print(f'  Embedding dimension:   {config["embedding_dim"]}')
print(f'  Training rounds:       {config["num_rounds"]}')
print(f'  Classifier:            {config["classifier"]}')
print(f'  Aggregation:           {config["aggregation_method"]}')

history = results['training_history']
print(f'\nTraining Progress:')
print('-' * 70)

for round_data in history:
    round_num = round_data['round']
    metrics = round_data['aggregated_metrics']
    loss = metrics['avg_loss']
    acc = metrics['avg_accuracy']
    print(f'  Round {round_num}: loss={loss:.6f}, accuracy={acc:.4f}')

first_loss = history[0]['aggregated_metrics']['avg_loss']
final_loss = history[-1]['aggregated_metrics']['avg_loss']
improvement = first_loss - final_loss
improvement_pct = (improvement / first_loss) * 100

print(f'\nResults Summary:')
print('-' * 70)
print(f'  Loss reduced by:     {improvement:.6f} ({improvement_pct:.2f}%)')
print(f'  Initial loss:        {first_loss:.6f}')
print(f'  Final loss:          {final_loss:.6f}')
print(f'  Status:              ✅ Complete!')
print(f'\n✅ All results saved in: federated_learning_results.json\n')
