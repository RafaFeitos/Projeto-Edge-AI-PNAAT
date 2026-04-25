# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **“Reopen in Container”**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`



### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Rafael Silva Arraes Feitosa

---

### ▶️ Como Executar

Execute os scripts nesta ordem a partir da raiz do repositório:

```bash
# 1. Treinamento — gera model.h5
python train_model.py

# 2. Otimização — gera model.tflite e model_float16.tflite
python optimize_model.py
```

---

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` tem dois blocos convolucionais seguidos de um classificador denso, projetada para ser simples e compatível com execução em dispositivos embarcados.

| Camada | Tipo | Configuração | Motivo |
|--------|------|-------------|--------|
| 1 | Conv2D | 32 filtros 3×3, ReLU | Identifica características básicas dos dígitos — bordas e traços |
| 2 | MaxPooling2D | 2×2 | Reduz dimensionalidade pela metade, mantendo as features relevantes |
| 3 | Conv2D | 64 filtros 3×3, ReLU | Identifica formas mais complexas combinando as features anteriores |
| 4 | MaxPooling2D | 2×2 | Segunda redução — tensor chega ao classificador com 5×5×64 valores |
| 5 | Flatten | — | Converte o tensor 3D em vetor 1D para entrada no classificador |
| 6 | Dense | 64 neurônios, ReLU | Aprende relações globais entre as features extraídas |
| 7 | Dense | 10 neurônios, Softmax | Uma probabilidade por dígito (0–9) |

**Total de parâmetros treináveis: 121.930**

A arquitetura usa 2 blocos Conv+Pool em vez de 3 porque o MNIST é um dataset relativamente simples — imagens 28×28 em escala de cinza com padrões regulares. Um terceiro bloco adicionaria custo computacional sem ganho real de acurácia nesse contexto.

---

### 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| TensorFlow | 2.21.0 | Construção, treinamento e conversão do modelo |
| Keras | 3.14.0 | Definição das camadas da CNN |
| os | stdlib | Leitura do tamanho dos arquivos gerados no comparativo |

---

### 3️⃣ Técnica de Otimização do Modelo

Foram aplicadas e comparadas duas técnicas de quantização em `optimize_model.py`:

**Técnica 1 — Dynamic Range Quantization (`model.tflite`)**

Converte os pesos de float32 para int8 em tempo de conversão. As ativações são quantizadas dinamicamente durante a inferência. Não exige dataset de calibração.

- Maior redução de tamanho entre as duas técnicas
- Compatível com qualquer CPU, incluindo microcontroladores sem unidade de ponto flutuante
- Indicada para dispositivos de baixo custo como ESP32 e STM32

**Técnica 2 — Float16 Quantization (`model_float16.tflite`)**

Converte os pesos de float32 para float16, mantendo ponto flutuante. Também não exige calibração.

- Redução menor que a Dynamic Range, porém com maior fidelidade numérica
- Preferível em dispositivos com acelerador dedicado float16, como Coral Edge TPU ou NVIDIA Jetson Nano

**Comparativo de tamanho:**

| Modelo | Tamanho | Redução |
|--------|---------|---------|
| model.h5 (float32 original) | 1.467,1 KB | referência |
| model.tflite (Dynamic Range) | 128,2 KB | 91,3% |
| model_float16.tflite (Float16) | 243,7 KB | 83,4% |

A Dynamic Range Quantization foi selecionada como técnica principal por oferecer maior compressão e compatibilidade ampla com hardware embarcado de baixo custo.

---

### 4️⃣ Resultados Obtidos

| Métrica | Valor |
|---------|-------|
| Acurácia final no conjunto de teste | 98,92% |
| Épocas de treinamento | 5 |
| Tamanho model.h5 | 1.467,1 KB |
| Tamanho model.tflite (Dynamic Range) | 128,2 KB |
| Tamanho model_float16.tflite (Float16) | 243,7 KB |
| Redução Dynamic Range | 91,3% |
| Redução Float16 | 83,4% |

O modelo atingiu 98,92% de acurácia com 5 épocas em CPU, confirmando que a arquitetura escolhida é suficiente para o problema sem necessidade de maior complexidade.

---

### 5️⃣ Comentários Adicionais

O desenvolvimento foi feito de forma incremental, com funções separadas e nomes claros para facilitar a leitura do código.

A seed `tf.random.set_seed(42)` foi adicionada para garantir reprodutibilidade dos resultados entre execuções e ambientes distintos — decisão importante para projetos avaliados por CI automatizado.

**Dificuldades encontradas:**

A principal dificuldade foi a incompatibilidade entre o Keras 3.x e a API anterior: o `input_shape` diretamente na camada Conv2D gera um warning nessa versão, mas não afeta o funcionamento. Durante a execução da pipeline, também foi necessário ajustar a configuração da secret `TOKEN` no repositório para permitir a validação correta do workflow no CI.

**Limitações do modelo:**

O modelo foi treinado exclusivamente com o MNIST — dígitos centralizados, fundo preto e traço branco. Em imagens do mundo real com ruído, rotação ou escala diferente, o desempenho pode ser inferior. Para aplicações embarcadas reais, seria necessário um dataset mais diverso e possivelmente técnicas de data augmentation.

---

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
