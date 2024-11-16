import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class SolanaAnalyzer:
    def __init__(self, symbol='SOL-USD'):
        self.symbol = symbol
        self.data = None

    def collect_data(self, start_date='2020-01-01'):
        """Coletar e processar dados históricos de preços e volumes da Solana."""
        self.data = yf.download(self.symbol, start=start_date)
        print("Dados coletados com sucesso.")

    def calculate_indicators(self):
        """Aplicar indicadores técnicos: MA, RSI, MACD e Bandas de Bollinger."""
        if self.data is None:
            raise Exception("Dados não carregados. Execute collect_data primeiro.")

        # Cálculo da Média Móvel
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()

        # Cálculo do Índice de Força Relativa (RSI)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Cálculo das Bandas de Bollinger
        self.data['20 Std'] = self.data['Close'].rolling(window=20).std()
        self.data['Upper Band'] = self.data['MA20'] + (self.data['20 Std'] * 2)
        self.data['Lower Band'] = self.data['MA20'] - (self.data['20 Std'] * 2)

        # Cálculo do MACD
        self.data['EMA12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
        self.data['Signal Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

        print("Indicadores calculados com sucesso.")

    def plot_data(self):
        """Gerar gráficos para visualização das tendências de preço e volume ao longo do tempo."""
        if self.data is None:
            raise Exception("Dados não carregados. Execute collect_data primeiro.")

        plt.figure(figsize=(14, 10))

        # Gráfico dos preços e Bandas de Bollinger
        plt.subplot(3, 1, 1)
        plt.plot(self.data[['Close', 'MA20', 'Upper Band', 'Lower Band']], label=['Preço', 'MA20', 'Banda Superior', 'Banda Inferior'])
        plt.title('Preço da Solana e Bandas de Bollinger')
        plt.legend()

        # Gráfico do RSI
        plt.subplot(3, 1, 2)
        plt.plot(self.data['RSI'], label='RSI', color='orange')
        plt.title('Índice de Força Relativa (RSI)')
        plt.axhline(70, linestyle='--', alpha=0.5, color='red')
        plt.axhline(30, linestyle='--', alpha=0.5, color='green')
        plt.legend()

        # Gráfico do MACD
        plt.subplot(3, 1, 3)
        plt.plot(self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data['Signal Line'], label='Signal Line', color='red')
        plt.title('MACD')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def alert_system(self):
        """Permitir que o usuário configure alertas para pontos de entrada e saída."""
        if self.data is None:
            raise Exception("Dados não carregados. Execute collect_data primeiro.")

        last_ma = self.data['MA20'].iloc[-1]
        last_price = self.data['Close'].iloc[-1]
        last_rsi = self.data['RSI'].iloc[-1]

        # Alertas para cruzamentos de média
        if last_price > last_ma:
            print(f"Aviso: O preço ({last_price:.2f}) cruzou acima da MA de 20 dias ({last_ma:.2f}). Considere comprar.")
        elif last_price < last_ma:
            print(f"Aviso: O preço ({last_price:.2f}) cruzou abaixo da MA de 20 dias ({last_ma:.2f}). Considere vender.")

        # Alertas para o RSI
        if last_rsi > 70:
            print(f"Aviso: O RSI está alto ({last_rsi:.2f}). Considere realizar vendas.")
        elif last_rsi < 30:
            print(f"Aviso: O RSI está baixo ({last_rsi:.2f}). Considere realizar compras.")

if __name__ == "__main__":
    analyzer = SolanaAnalyzer()
    analyzer.collect_data()
    analyzer.calculate_indicators()
    analyzer.plot_data()
    analyzer.alert_system()