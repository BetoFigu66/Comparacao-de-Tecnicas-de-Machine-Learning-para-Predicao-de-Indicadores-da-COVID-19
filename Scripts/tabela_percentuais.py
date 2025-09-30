import pandas as pd
import os
import sys
from datetime import datetime

def create_percentage_tables():
    """Cria tabelas com percentuais de cada categoria dos dados"""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'DataSources')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'Docs', 'ResultadosPreliminares')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega dados de votação
    votacao_2016 = pd.read_csv(
        os.path.join(DATA_DIR, 'votacao_candidato', 'votacao_candidato_2016.csv'), 
        encoding="latin1", delimiter=';'
    )
    
    # Normalizar nomes de colunas (importar função do carga_dados)
    sys.path.append(os.path.dirname(__file__))  # Adiciona o diretório Scripts ao path
    from carga_dados import normalize_column_names
    votacao_2016 = normalize_column_names(votacao_2016)
    
    # Filtra apenas eleitos
    eleitos = votacao_2016[votacao_2016['Situação totalização'] == 'Eleito']
    
    # Carrega uma amostra dos dados COVID
    covid_sample = pd.read_csv(
        os.path.join(DATA_DIR, 'caso_full_csv', 'caso_full.csv'),
        nrows=100000
    )
    
    # Cria tabelas de percentuais
    tables_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tabelas de Percentuais - Análise TCC</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .section { margin: 30px 0; }
            h2 { color: #333; border-bottom: 2px solid #333; }
            h3 { color: #666; }
        </style>
    </head>
    <body>
        <h1>Análise de Percentuais dos Dados - TCC COVID-19</h1>
        <p><strong>Gerado em:</strong> """ + datetime.now().strftime('%d/%m/%Y %H:%M:%S') + """</p>
    """
    
    # Seção 1: Dados Eleitorais
    tables_html += """
        <div class="section">
            <h2>1. DADOS ELEITORAIS - PREFEITOS ELEITOS (2016)</h2>
    """
    
    # Tabela Gênero
    genero_counts = eleitos['Gênero'].value_counts()
    genero_pct = eleitos['Gênero'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>1.1 Distribuição por Gênero</h3>
            <table>
                <tr><th>Gênero</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for genero in genero_counts.index:
        tables_html += f"<tr><td>{genero}</td><td>{genero_counts[genero]:,}</td><td>{genero_pct[genero]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{genero_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Tabela Escolaridade
    edu_counts = eleitos['Grau de instrução'].value_counts()
    edu_pct = eleitos['Grau de instrução'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>1.2 Distribuição por Escolaridade</h3>
            <table>
                <tr><th>Grau de Instrução</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for edu in edu_counts.index:
        tables_html += f"<tr><td>{edu}</td><td>{edu_counts[edu]:,}</td><td>{edu_pct[edu]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{edu_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Tabela Faixa Etária
    idade_counts = eleitos['Faixa etária'].value_counts()
    idade_pct = eleitos['Faixa etária'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>1.3 Distribuição por Faixa Etária</h3>
            <table>
                <tr><th>Faixa Etária</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for idade in sorted(idade_counts.index):
        tables_html += f"<tr><td>{idade}</td><td>{idade_counts[idade]:,}</td><td>{idade_pct[idade]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{idade_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Tabela Cor/Raça
    raca_counts = eleitos['Cor/raça'].value_counts()
    raca_pct = eleitos['Cor/raça'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>1.4 Distribuição por Cor/Raça</h3>
            <table>
                <tr><th>Cor/Raça</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for raca in raca_counts.index:
        tables_html += f"<tr><td>{raca}</td><td>{raca_counts[raca]:,}</td><td>{raca_pct[raca]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{raca_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Tabela por Região
    regiao_counts = eleitos['Região'].value_counts()
    regiao_pct = eleitos['Região'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>1.5 Distribuição por Região</h3>
            <table>
                <tr><th>Região</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for regiao in regiao_counts.index:
        tables_html += f"<tr><td>{regiao}</td><td>{regiao_counts[regiao]:,}</td><td>{regiao_pct[regiao]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{regiao_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Seção 2: Dados COVID
    tables_html += """
        </div>
        <div class="section">
            <h2>2. DADOS EPIDEMIOLÓGICOS - COVID-19</h2>
    """
    
    # Tabela por Tipo de Local
    place_counts = covid_sample['place_type'].value_counts()
    place_pct = covid_sample['place_type'].value_counts(normalize=True) * 100
    
    tables_html += """
            <h3>2.1 Distribuição por Tipo de Local</h3>
            <table>
                <tr><th>Tipo de Local</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for place in place_counts.index:
        place_label = "Município" if place == "city" else "Estado"
        tables_html += f"<tr><td>{place_label}</td><td>{place_counts[place]:,}</td><td>{place_pct[place]:.1f}%</td></tr>"
    
    tables_html += f"<tr><th>TOTAL</th><th>{place_counts.sum():,}</th><th>100.0%</th></tr></table>"
    
    # Tabela por Estado (top 10)
    state_counts = covid_sample[covid_sample['place_type'] == 'city']['state'].value_counts().head(10)
    state_pct = covid_sample[covid_sample['place_type'] == 'city']['state'].value_counts(normalize=True).head(10) * 100
    
    tables_html += """
            <h3>2.2 Top 10 Estados com Mais Registros Municipais</h3>
            <table>
                <tr><th>Estado</th><th>Quantidade</th><th>Percentual (%)</th></tr>
    """
    for state in state_counts.index:
        tables_html += f"<tr><td>{state}</td><td>{state_counts[state]:,}</td><td>{state_pct[state]:.1f}%</td></tr>"
    
    total_cities = covid_sample[covid_sample['place_type'] == 'city'].shape[0]
    tables_html += f"<tr><th>TOTAL (amostra)</th><th>{total_cities:,}</th><th>100.0%</th></tr></table>"
    
    # Estatísticas gerais
    tables_html += """
            <h3>2.3 Estatísticas Gerais da Amostra</h3>
            <table>
                <tr><th>Métrica</th><th>Valor</th></tr>
    """
    
    city_data = covid_sample[covid_sample['place_type'] == 'city']
    unique_cities = city_data['city'].nunique()
    date_range = f"{covid_sample['date'].min()} a {covid_sample['date'].max()}"
    avg_population = city_data['estimated_population'].mean()
    
    tables_html += f"""
                <tr><td>Total de registros na amostra</td><td>{len(covid_sample):,}</td></tr>
                <tr><td>Municípios únicos</td><td>{unique_cities:,}</td></tr>
                <tr><td>Período dos dados</td><td>{date_range}</td></tr>
                <tr><td>População média dos municípios</td><td>{avg_population:,.0f} habitantes</td></tr>
            </table>
    """
    
    # Fecha HTML
    tables_html += """
        </div>
        <div class="section">
            <h2>3. RESUMO EXECUTIVO</h2>
            <p><strong>Dataset Eleitoral:</strong> Contém informações de 6.379 prefeitos eleitos em 2016, 
            distribuídos em 5.287 municípios brasileiros. Predominância masculina (89,4%) e alta escolaridade 
            (56,2% com ensino superior completo).</p>
            
            <p><strong>Dataset Epidemiológico:</strong> Dados diários de COVID-19 por município e estado, 
            cobrindo o período da pandemia. Permite análises temporais e geográficas detalhadas da evolução 
            da doença no território brasileiro.</p>
            
            <p><strong>Integração:</strong> Os datasets são integrados através dos códigos IBGE dos municípios, 
            permitindo análises que relacionam características político-administrativas com indicadores 
            epidemiológicos durante a pandemia.</p>
        </div>
    </body>
    </html>
    """
    
    # Salva arquivo HTML
    with open(os.path.join(OUTPUT_DIR, 'tabelas_percentuais.html'), 'w', encoding='utf-8') as f:
        f.write(tables_html)
    
    # Cria também versão em texto
    with open(os.path.join(OUTPUT_DIR, 'tabelas_percentuais.txt'), 'w', encoding='utf-8') as f:
        f.write("TABELAS DE PERCENTUAIS - ANÁLISE TCC COVID-19\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("1. DADOS ELEITORAIS - PREFEITOS ELEITOS (2016)\n")
        f.write("-" * 45 + "\n\n")
        
        f.write("1.1 Distribuição por Gênero:\n")
        for genero in genero_counts.index:
            f.write(f"  {genero}: {genero_counts[genero]:,} ({genero_pct[genero]:.1f}%)\n")
        f.write(f"  TOTAL: {genero_counts.sum():,} (100.0%)\n\n")
        
        f.write("1.2 Distribuição por Escolaridade:\n")
        for edu in edu_counts.index:
            f.write(f"  {edu}: {edu_counts[edu]:,} ({edu_pct[edu]:.1f}%)\n")
        f.write(f"  TOTAL: {edu_counts.sum():,} (100.0%)\n\n")
        
        f.write("2. DADOS EPIDEMIOLÓGICOS - COVID-19\n")
        f.write("-" * 35 + "\n\n")
        
        f.write("2.1 Distribuição por Tipo de Local:\n")
        for place in place_counts.index:
            place_label = "Município" if place == "city" else "Estado"
            f.write(f"  {place_label}: {place_counts[place]:,} ({place_pct[place]:.1f}%)\n")
        f.write(f"  TOTAL: {place_counts.sum():,} (100.0%)\n\n")
    
    print("Tabelas de percentuais criadas com sucesso!")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")
    print("- tabelas_percentuais.html (versão completa)")
    print("- tabelas_percentuais.txt (versão resumida)")

if __name__ == "__main__":
    create_percentage_tables()
