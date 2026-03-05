# =============================================================================
# cell 28: V15 SpikingBrain Visualizations
# =============================================================================

if not MATPLOTLIB_AVAILABLE:
    print("matplotlib unavailable: skipping V15 SpikingBrain visualizations.")
else:
    def plot_firing_rate_histogram(
        firing_rates: np.ndarray,
        target_rate: float = 0.38,
        title: str = 'Firing Rate Distribution',
        show: bool = True,
    ):
        """Plot histogram of per-channel firing rates."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(firing_rates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(target_rate, color='red', linestyle='--', linewidth=2, label=f'Target: {target_rate}')
        ax.axvline(firing_rates.mean(), color='orange', linestyle='-', linewidth=2, label=f'Mean: {firing_rates.mean():.3f}')

        # Healthy range shading
        ax.axvspan(0.2, 0.6, alpha=0.1, color='green', label='Healthy range [0.2, 0.6]')

        ax.set_xlabel('Firing Rate')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/figures/v15_firing_rate_dist.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


    def plot_cka_by_layer(
        cka_values: Dict[str, float],
        threshold: float = 0.3,
        title: str = 'CKA Similarity by Layer',
        show: bool = True,
    ):
        """Plot CKA similarity as a bar chart."""
        # Filter to per-layer values only
        layer_cka = {k: v for k, v in cka_values.items() if 'layer_' in k and 'mean' not in k}

        if not layer_cka:
            print("No per-layer CKA values to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(layer_cka.keys())
        values = [layer_cka[n] for n in names]
        colors = ['green' if v >= threshold else 'red' for v in values]

        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        ax.axhline(cka_values.get('cka_mean', 0), color='blue', linestyle='-', linewidth=2, label=f'Mean: {cka_values.get("cka_mean", 0):.3f}')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('layer_', 'L').replace('_to_', '->') for n in names], rotation=45, ha='right')
        ax.set_ylabel('CKA Similarity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/figures/v15_cka_by_layer.png', dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


    # Collect all firing rates from health metrics
    all_rates = []
    for key, rates in v15_results.health.per_channel_rates.items():
        if len(rates) > 0:
            all_rates.append(rates)

    if all_rates:
        combined_rates = np.concatenate(all_rates)

        # Firing rate histogram
        plot_firing_rate_histogram(
            firing_rates=combined_rates,
            target_rate=0.38,
            title=f'V15 Firing Rate Distribution (mean={combined_rates.mean():.3f})',
            show=True,
        )

    # CKA by layer
    if v15_results.cka:
        plot_cka_by_layer(
            cka_values=v15_results.cka,
            threshold=0.3,
            title='V15 CKA Similarity: Spikes vs Teacher',
            show=True,
        )

    print(f'\nVisualizations saved to {OUTPUT_DIR}/figures/')
