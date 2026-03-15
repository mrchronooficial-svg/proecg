# Landing Page ProECG — Estrutura & Copy para Implementação

> **Objetivo:** Vender assinaturas direto. Converter médicos visitantes em assinantes pagantes.
> **Público-alvo:** Médicos plantonistas de emergência, UTI e UBS no Brasil.
> **Tom:** Profissional-médico, confiável, direto. Sem ser frio — transmitir modernidade e agilidade.
> **Idioma:** Português brasileiro.
> **Abordagem:** Mobile-first (médico acessa pelo celular no plantão).

---

## PALETA DE CORES

```
--color-text:        #122056    /* Navy escuro — todo texto principal */
--color-accent:      #5B65DC    /* Azul accent — botões primários, links, destaques */
--color-secondary:   #EEEFFD    /* Azul claro — cards, badges, backgrounds de seção alternada */
--color-bg:          #FAFAFD    /* Fundo principal da página */
--color-white:       #FFFFFF    /* Cards, navbar, seções brancas */
--color-accent-hover: #4A51C5   /* Hover nos botões (accent um tom mais escuro) */
--color-text-light:  #4A5078    /* Texto secundário, subtítulos, descrições */
--color-success:     #10B981    /* Verde — badge de destaque no plano recomendado */
--color-warning:     #F59E0B    /* Amarelo — disclaimer/aviso legal */
--color-border:      #E2E4F0    /* Bordas sutis de cards */
```

## TIPOGRAFIA

```
--font-display: 'Satoshi', 'General Sans', ou 'Cabinet Grotesk'  /* Headlines — bold, moderno, médico */
--font-body: 'Inter' ou 'DM Sans'                                 /* Body — alta legibilidade */
--font-mono: 'JetBrains Mono'                                     /* Dados técnicos do laudo (FC, QRS, etc.) */

/* Tamanhos Desktop */
H1: 56px / font-weight: 800 / line-height: 1.1
H2: 40px / font-weight: 700 / line-height: 1.2
H3: 24px / font-weight: 600 / line-height: 1.3
Body large: 20px / font-weight: 400 / line-height: 1.6
Body: 16px / font-weight: 400 / line-height: 1.6
Small/caption: 14px / font-weight: 500 / line-height: 1.4

/* Tamanhos Mobile */
H1: 36px
H2: 28px
H3: 20px
Body large: 18px
Body: 16px
```

## ANIMAÇÕES

```
/* Todas as seções entram com fade-in + slide-up ao scrollar (Intersection Observer ou Framer Motion) */
--animation-reveal: fadeInUp 0.6s ease-out
--animation-delay-step: 0.1s  /* Stagger entre elementos da mesma seção */

/* Hero: elementos entram em sequência — headline → subtítulo → CTA → mockup */
/* Logo bar: marquee scroll horizontal infinito */
/* Cards de features: hover com scale(1.02) + shadow increase */
/* Botões: hover com translateY(-2px) + shadow */
/* Mockup do ECG: pulso de brilho sutil (glow pulse) para chamar atenção */
/* Números na social proof: counter animation (0 → número final) */
/* Pricing cards: plano destacado com escala levemente maior (scale 1.05) e borda accent */
```

---

## ESTRUTURA DAS SEÇÕES

---

### SEÇÃO 0 — NAVBAR (sticky)

**`[Layout]`** Fixo no topo. Background `--color-white` com `backdrop-filter: blur(12px)` e opacidade 95%. Sombra sutil no scroll. Max-width 1200px centralizado. Altura ~64px.

**`[Logo]`** ProECG — à esquerda. Ícone estilizado de onda ECG + texto "ProECG". Cor `--color-text`.

**`[Links — desktop only, hidden no mobile]`**
- Como Funciona
- Benefícios
- Planos
- FAQ

**`[CTAs — direita]`**
- `[Link texto]` Entrar
- `[Botão primário accent]` Assinar Agora

**`[Mobile]`** Hamburger menu. Ao abrir: links + CTAs em coluna. Botão "Assinar Agora" full-width.

---

### SEÇÃO 1 — HERO

**`[Layout]`** Centralizado. Padding vertical generoso (120px top, 80px bottom desktop / 80px top, 60px bottom mobile). Background `--color-bg`. Conteúdo max-width 800px para texto, mockup pode ir até 1200px.

**`[Badge — acima da headline]`**
```
🧠 Laudo de ECG com IA em segundos
```
Formato: pill/badge com background `--color-secondary`, texto `--color-accent`, font-size 14px, border-radius 999px, padding 6px 16px. Ícone de cérebro ou raio à esquerda.

**`[H1 — Headline principal]`**
```
Fotografou o ECG? O laudo já saiu.
```
Centralizado. 56px desktop, 36px mobile. Cor `--color-text`. Peso 800. Max-width 700px.

> **Nota de copy:** Headline de 6 palavras. Estrutura de pergunta+resposta que transmite velocidade. Usa linguagem coloquial médica ("fotografou"). O gatilho é VELOCIDADE — a principal dor do plantonista.

**`[Subtítulo — body large]`**
```
Tire uma foto do ECG de papel com seu celular e receba um laudo descritivo completo com hipóteses diagnósticas — validado por cardiologistas, direto no seu bolso.
```
Centralizado. 20px desktop, 18px mobile. Cor `--color-text-light`. Max-width 640px. Peso 400.

> **Nota de copy:** Expande a headline: O QUÊ (foto do ECG → laudo), COMO (celular), QUALIDADE (validado por cardiologistas), ONDE (no seu bolso = mobile). Gatilhos: praticidade + autoridade médica.

**`[CTA — botão primário]`**
```
Começar Agora — R$ 197/mês
```
Background `--color-accent`. Texto branco. Font-size 18px, peso 600. Padding 16px 40px. Border-radius 12px. Sombra: `0 4px 14px rgba(91,101,220,0.35)`. Hover: `translateY(-2px)` + sombra maior.

> **Nota de copy:** CTA com preço do plano mais barato embutido. "Começar Agora" é ação imediata. O preço ancora na opção mais vantajosa. Sem "grátis" — o produto é premium e o público é médico com poder aquisitivo.

**`[Sub-CTA — texto abaixo do botão]`**
```
Pix ou cartão · Cancele quando quiser · Acesso imediato
```
Font-size 14px. Cor `--color-text-light`. Centralizado. Reduz objeções: forma de pagamento, compromisso, tempo de ativação.

**`[Elemento visual — Mockup]`**
À direita da copy (desktop) ou abaixo (mobile). Mockup de celular mostrando:
- Lado esquerdo: foto de um ECG de papel (levemente rotacionada, como se tirasse do celular)
- Seta animada (→ ou fluxo visual)
- Lado direito: tela do app com o laudo gerado

O mockup deve mostrar dados reais do laudo:
```
Ritmo: Sinusal | FC: 82 bpm
Eixo: +45° | PR: 168ms | QRS: 86ms
QT: 372ms | QTc: 418ms

Achados: Supradesnivelamento de ST em V1-V4
Sugestivo de: SCA com supra em parede anterior

⚠️ Ferramenta de apoio — correlacionar com clínica
```
Usar `--font-mono` para os dados do laudo no mockup. Efeito de glow sutil ao redor do mockup (`box-shadow` com `--color-accent` em 10% opacity).

---

### SEÇÃO 2 — SOCIAL PROOF (Números)

**`[Layout]`** Full-width. Background `--color-white`. Padding vertical 60px. Border-top e border-bottom em `--color-border`.

**`[Headline — opcional, pode ser omitida]`**
```
A confiança de quem já usa
```
Centralizado. H3 (24px). Cor `--color-text-light`.

**`[Métricas — grid de 3 colunas]`**

| Número | Label |
|--------|-------|
| `+2.500` | Laudos gerados |
| `< 30s` | Tempo médio do laudo |
| `~30` | Diagnósticos cobertos |

Números em `--color-text`, font-size 48px desktop / 36px mobile, peso 800. Counter animation ao entrar no viewport.
Labels em `--color-text-light`, font-size 16px, peso 400.

> **Nota:** Esses números são iniciais e devem ser atualizados conforme o produto cresce. Se ainda não houver dados reais, usar projeções ou dados de teste validados. Alternativamente, usar "30+ diagnósticos", "Laudo em segundos", "Validado por cardiologistas" como métricas qualitativas.

---

### SEÇÃO 3 — COMO FUNCIONA (3 passos)

**`[Layout]`** Background `--color-bg`. Padding vertical 100px. Max-width 1200px.

**`[H2]`**
```
Simples como tirar uma foto
```

> **Nota de copy:** 6 palavras. Referência ao gesto mais natural do celular. Gatilho de SIMPLICIDADE — crítico para médicos no plantão que não têm tempo para aprender ferramenta nova.

**`[Subtítulo]`**
```
Do ECG de papel ao laudo completo em 3 passos.
```
20px. Cor `--color-text-light`. Centralizado.

**`[3 cards em row (desktop) / stack (mobile)]`**

Cada card: background `--color-white`, border-radius 16px, padding 32px, sombra sutil. Border-top 4px `--color-accent`.

**Card 1:**
```
Ícone: 📸 (ou ícone de câmera estilizado)
Número: 01
Título: Fotografe
Descrição: Abra o ProECG no celular e tire uma foto do ECG de papel. Pode ser traçado de 12 derivações, tira de ritmo ou ECG portátil.
```

**Card 2:**
```
Ícone: ⚡ (ou ícone de processamento/IA)
Número: 02
Título: IA Analisa
Descrição: Nossa IA digitaliza o traçado, mede todos os intervalos (FC, PR, QRS, QT, QTc, eixo) e cruza com critérios clínicos validados por cardiologistas.
```

**Card 3:**
```
Ícone: 📋 (ou ícone de documento/laudo)
Número: 03
Título: Receba o Laudo
Descrição: Em segundos, você recebe o laudo descritivo completo com medições, achados e hipóteses diagnósticas. Exporte em PDF ou envie por WhatsApp.
```

> **Nota de design:** Os números (01, 02, 03) devem ser grandes e em `--color-accent` com opacidade reduzida (marca d'água atrás do card) ou como badge no topo. Conectar os cards com uma linha pontilhada horizontal (desktop) para transmitir fluxo/pipeline.

---

### SEÇÃO 4 — BENEFÍCIOS (Por que o ProECG?)

**`[Layout]`** Background `--color-white`. Padding vertical 100px. Layout zig-zag: imagem-esquerda/texto-direita alternando.

**`[H2]`**
```
Por que médicos escolhem o ProECG
```

> **Nota de copy:** "Médicos escolhem" = social proof implícita + identificação profissional.

---

**`[Bloco 1 — Velocidade]`**
Layout: Visual à esquerda, texto à direita.

```
[H3] Laudo em segundos, não em horas
[Body] No plantão, cada minuto conta. O ProECG entrega um laudo descritivo completo antes mesmo de você terminar de preencher o prontuário. Sem esperar parecer, sem ligar para o cardiologista de madrugada.
[Visual] Animação de cronômetro: "00:00" → "00:12" (12 segundos) com o laudo aparecendo ao lado.
```

> **Gatilhos:** Velocidade + dor real (esperar cardiologista de madrugada). Empatia com a rotina do plantonista.

---

**`[Bloco 2 — Abrangência Diagnóstica]`**
Layout: Texto à esquerda, visual à direita.

```
[H3] ~30 diagnósticos na ponta dos dedos
[Body] De SCA com supra a Brugada, de FA a bloqueios atrioventriculares — nosso motor cobre os diagnósticos que mais importam na emergência. Critérios clínicos definidos e validados por cardiologistas.
[Visual] Grid/lista visual dos principais diagnósticos cobertos, organizados por categoria:

Isquêmicos: SCA com/sem supra, equivalentes (Wellens, Winter, de Winter, Sgarbossa)
Arritmias: FA, Flutter, TSV, TV, Torsades, WPW
Bloqueios: BRD, BRE, BAV 1º/2º/3º
Outros: Brugada, QT longo, Hipercalemia, Pericardite, TEP
```

> **Nota de design:** Usar chips/tags coloridas por categoria (isquêmicos em vermelho suave, arritmias em laranja, bloqueios em azul, outros em roxo). Fundo `--color-secondary`.

---

**`[Bloco 3 — Confiança Clínica]`**
Layout: Visual à esquerda, texto à direita.

```
[H3] IA + olhar do cardiologista
[Body] Nosso motor combina algoritmos de digitalização, medições automáticas, regras clínicas baseadas em critérios consagrados e classificação por rede neural — tudo construído e validado em parceria com cardiologistas. Não é chatbot: é pipeline médico com rigor clínico.
[Visual] Diagrama simplificado do pipeline:
Foto → Digitalização → Medições → Regras Clínicas + CNN → Laudo
(Usar ícones conectados por setas, estilo flowchart minimalista)
```

> **Gatilhos:** Autoridade ("cardiologistas"), diferenciação ("não é chatbot"), rigor técnico sem ser inacessível.

---

**`[Bloco 4 — Mobile-First / Praticidade]`**
Layout: Texto à esquerda, visual à direita.

```
[H3] Feito para o bolso do jaleco
[Body] Acesse direto do celular, sem instalar nada. Funciona no navegador — abra, fotografe, receba o laudo. Exporte em PDF, compartilhe por WhatsApp com a equipe ou cole no prontuário.
[Visual] Mockup de celular na mão (contexto médico), mostrando a tela do dashboard com o botão "Novo ECG".
```

> **Gatilhos:** "Bolso do jaleco" = linguagem do médico. Sem instalar = zero atrito. WhatsApp = workflow real.

---

### SEÇÃO 5 — EXEMPLO DE LAUDO (Show, don't tell)

**`[Layout]`** Background `--color-secondary` (azul claro). Padding vertical 100px. Centralizado.

**`[H2]`**
```
Veja um laudo real
```

**`[Subtítulo]`**
```
Esse é o resultado que você recebe em segundos após fotografar o ECG.
```

**`[Card do laudo — centralizado, max-width 600px]`**
Background `--color-white`. Border-radius 16px. Padding 32px. Sombra média. Borda esquerda 4px `--color-accent`.

```
[Fonte mono — simular output do sistema]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAUDO ECG — ProECG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ritmo: Sinusal
Frequência cardíaca: 78 bpm
Eixo elétrico: +60°

Intervalos:
  PR: 180ms | QRS: 88ms
  QT: 380ms | QTc: 420ms (Bazett)

Achados:
  Supradesnivelamento do segmento ST
  em V1-V4 (>2mm)

Hipótese diagnóstica:
  Sugestivo de Síndrome Coronariana
  Aguda com supra de ST em parede
  anterior.

⚠️ Ferramenta de apoio à decisão
clínica — não substitui avaliação médica.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**`[Botões abaixo do card]`**
- `[Botão secundário]` Exportar PDF
- `[Botão secundário]` Enviar por WhatsApp

> **Nota de design:** Os botões são ilustrativos (não funcionais na LP). Mostram que o fluxo de exportação existe. Abaixo deles, texto pequeno: "Laudo gerado em 12 segundos a partir de foto de ECG de papel."

> **Gatilho:** "Show, don't tell" — o médico vê exatamente o que vai receber. Remove incerteza e gera desejo.

---

### SEÇÃO 6 — PRICING

**`[Layout]`** Background `--color-bg`. Padding vertical 100px. Max-width 1000px centralizado.

**`[H2]`**
```
Escolha seu plano
```

**`[Subtítulo]`**
```
Acesso ilimitado a laudos de ECG. Todos os planos incluem todas as funcionalidades.
```
Cor `--color-text-light`. Centralizado.

> **Nota de copy:** "Acesso ilimitado" + "todas as funcionalidades" = sem surpresas, sem feature-gating. Médico quer saber que está pagando por tudo.

**`[3 cards de pricing em row (desktop) / stack (mobile)]`**

Todos com: background `--color-white`, border-radius 16px, padding 32px, sombra sutil.

---

**Card 1 — Mensal:**
```
[Nome do plano] Mensal
[Preço] R$ 267 /mês
[Detalhe] Cobrança mensal
[Descrição] Ideal para testar o ProECG sem compromisso longo.
[Lista de features]
  ✓ Laudos ilimitados
  ✓ ~30 diagnósticos
  ✓ Exportação PDF
  ✓ Compartilhamento WhatsApp
  ✓ Histórico de 30 dias
  ✓ Suporte por email
[CTA — botão outline/secundário] Assinar Mensal
```
Borda: `--color-border`. Sem destaque especial.

---

**Card 2 — Semestral (DESTAQUE):**
```
[Badge topo] 🔥 Mais popular
[Nome do plano] Semestral
[Preço] R$ 227 /mês
[Detalhe] Cobrança a cada 6 meses (R$ 1.362 total)
[Economia] Economize R$ 240/ano vs. mensal
[Descrição] O equilíbrio perfeito entre economia e flexibilidade.
[Lista de features]
  ✓ Tudo do plano Mensal
  ✓ 15% de desconto
  ✓ Suporte prioritário
[CTA — botão primário accent] Assinar Semestral
```
Card com: `border: 2px solid var(--color-accent)`, `scale(1.05)` em desktop, badge `--color-success` no topo. Sombra mais forte que os outros.

---

**Card 3 — Anual:**
```
[Badge topo] 💰 Melhor preço
[Nome do plano] Anual
[Preço] R$ 197 /mês
[Detalhe] Cobrança anual (R$ 2.364 total)
[Economia] Economize R$ 840/ano vs. mensal
[Descrição] O menor preço por mês para quem já decidiu.
[Lista de features]
  ✓ Tudo do plano Semestral
  ✓ 26% de desconto
  ✓ Suporte prioritário
[CTA — botão outline/secundário] Assinar Anual
```
Badge `--color-warning` (amarelo/dourado).

---

**`[Texto abaixo dos cards — centralizado]`**
```
Pagamento via Pix ou cartão de crédito · Processado com segurança pelo Asaas
Cancele quando quiser · Sem multa, sem burocracia
```
Font-size 14px. Cor `--color-text-light`.

> **Nota de design:** O card do meio (Semestral) deve estar visualmente elevado — usar scale, borda accent, e badge "Mais popular" para guiar a decisão. Efeito ancoragem: o olho vê R$267, depois R$227, e R$197 parece a barganha.

---

### SEÇÃO 7 — CONFIANÇA & COMPLIANCE

**`[Layout]`** Background `--color-white`. Padding vertical 80px. Grid de 2x2 (desktop) / stack (mobile). Max-width 1000px.

**`[H2]`**
```
Segurança e conformidade
```

**`[4 blocos com ícone + título + descrição]`**

**Bloco 1:**
```
Ícone: 🔒 (cadeado)
Título: Dados Protegidos (LGPD)
Descrição: Nenhum dado identificável de paciente é coletado ou armazenado. Apenas a foto do ECG é processada. Conformidade total com a LGPD.
```

**Bloco 2:**
```
Ícone: 🩺 (estetoscópio)
Título: Validado por Cardiologistas
Descrição: Todos os critérios diagnósticos e regras clínicas foram definidos e revisados por cardiologistas com experiência em eletrocardiografia.
```

**Bloco 3:**
```
Ícone: ⚠️ (alerta)
Título: Ferramenta de Apoio
Descrição: O ProECG é uma ferramenta de apoio à decisão clínica. Não substitui o julgamento médico. Cada laudo inclui disclaimer obrigatório.
```

**Bloco 4:**
```
Ícone: 🗑️ (expiração)
Título: Dados Temporários
Descrição: Análises expiram automaticamente após 30 dias. Exporte em PDF antes se quiser manter o registro.
```

> **Gatilhos:** Segurança + transparência + conformidade regulatória. Médicos se preocupam com implicações legais — essa seção antecipa e resolve.

---

### SEÇÃO 8 — TESTIMONIALS / DEPOIMENTOS

**`[Layout]`** Background `--color-secondary`. Padding vertical 100px. Carrossel horizontal com auto-scroll.

**`[H2]`**
```
O que médicos estão dizendo
```

> **Nota:** No lançamento, se ainda não houver depoimentos reais, usar depoimentos de beta testers ou do cardiologista parceiro. Marcar como "beta tester" ou "consultor clínico". NUNCA usar depoimentos falsos.

**`[Cards de depoimentos — carrossel]`**

Cada card: background `--color-white`, border-radius 16px, padding 24px, max-width 400px. Ícone de aspas grande em `--color-accent` com 10% opacity no background.

**Estrutura de cada card:**
```
[Aspas decorativas — " "]
[Texto do depoimento — 2-3 frases]
[Separador — linha fina]
[Foto circular] [Nome] [Especialidade, CRM]
```

**Depoimento exemplo 1 (usar como template):**
```
"Uso o ProECG nos plantões de emergência. Antes eu ficava inseguro com ECGs complexos de madrugada — agora tenho um segundo olhar em segundos."
— Dr. [Nome], Emergencista, CRM-RJ [xxxxx]
```

**Depoimento exemplo 2:**
```
"A digitalização é impressionante. Ele mede os intervalos com precisão e já sugere o diagnóstico. Economizo tempo e ganho segurança."
— Dra. [Nome], Clínica Geral (UBS), CRM-SP [xxxxx]
```

**Depoimento exemplo 3 (cardiologista parceiro):**
```
"Participei da construção dos critérios diagnósticos. O rigor clínico do motor é o mesmo que eu uso na minha prática diária."
— Dr. [Nome], Cardiologista, CRM-[XX] [xxxxx]
```

> **Nota importante:** Substituir pelos depoimentos reais assim que disponíveis. O CRM é crucial para credibilidade médica.

---

### SEÇÃO 9 — FAQ

**`[Layout]`** Background `--color-bg`. Padding vertical 100px. Max-width 800px centralizado. Accordion (clica na pergunta, expande a resposta).

**`[H2]`**
```
Perguntas frequentes
```

**`[Accordion — 8 perguntas]`**

**P1:** O ProECG substitui o cardiologista?
```
Não. O ProECG é uma ferramenta de apoio à decisão clínica. Ele auxilia na interpretação, mas o diagnóstico definitivo e a conduta terapêutica são sempre responsabilidade do médico assistente. Cada laudo inclui um disclaimer obrigatório.
```

**P2:** Quais tipos de ECG são aceitos?
```
ECGs de 12 derivações em papel, tiras de ritmo e traçados de aparelhos portáteis. Basta fotografar com o celular — o sistema aceita fotos em qualquer ângulo e iluminação razoáveis.
```

**P3:** Quantos diagnósticos o sistema cobre?
```
Atualmente ~30 diagnósticos, incluindo SCA com e sem supra, fibrilação atrial, flutter, taquicardias ventriculares, bloqueios de ramo e AV, síndromes de Brugada e Wellens, hipercalemia, pericardite, entre outros. Novos diagnósticos são adicionados continuamente.
```

**P4:** Os dados dos meus pacientes ficam armazenados?
```
Não coletamos nenhum dado identificável de pacientes — sem nome, CPF ou informações pessoais. Apenas a foto do ECG é processada. As análises ficam disponíveis por 30 dias e são automaticamente excluídas após esse período.
```

**P5:** Preciso instalar algum aplicativo?
```
Não. O ProECG funciona 100% no navegador do celular. Basta acessar o site, fazer login e começar a usar. Funciona em qualquer smartphone com câmera.
```

**P6:** Como funciona o pagamento?
```
Aceitamos Pix e cartão de crédito, processados com segurança pelo Asaas. Você escolhe entre plano mensal, semestral ou anual. O acesso é liberado imediatamente após a confirmação do pagamento.
```

**P7:** Posso cancelar a qualquer momento?
```
Sim. Não há multa nem burocracia. Cancele quando quiser diretamente pelo painel. Você mantém o acesso até o final do período já pago.
```

**P8:** O laudo tem validade legal?
```
O ProECG gera um laudo de apoio, não um laudo médico formal assinado. Ele serve como auxílio para a interpretação clínica do médico assistente, que é o responsável pelo diagnóstico e pela conduta.
```

> **Nota de design:** Accordion com ícone de + / − à direita. Ao expandir, transição suave (height auto + opacity). Pergunta em peso 600, resposta em peso 400 cor `--color-text-light`.

---

### SEÇÃO 10 — CTA FINAL

**`[Layout]`** Background `--color-text` (navy escuro #122056 — inversão). Padding vertical 100px. Texto em branco.

**`[H2]`**
```
O próximo ECG não precisa te deixar em dúvida.
```
Cor: branco. Centralizado. Max-width 600px.

> **Nota de copy:** Fala direto com a dor emocional (dúvida na interpretação, insegurança no plantão). Gatilho de DOR → SOLUÇÃO imediata logo abaixo.

**`[Subtítulo]`**
```
Comece agora e tenha um cardiologista de IA no bolso do jaleco.
```
Cor: branco com 80% opacidade. Centralizado.

**`[CTA — botão branco com texto accent]`**
```
Assinar o ProECG
```
Background: branco. Texto: `--color-accent`. Font-size 18px, peso 600. Padding 16px 40px. Border-radius 12px. Hover: background levemente cinza.

**`[Sub-texto]`**
```
A partir de R$ 197/mês · Pix ou cartão · Acesso imediato
```
Cor: branco com 60% opacidade. Font-size 14px.

---

### SEÇÃO 11 — FOOTER

**`[Layout]`** Background `--color-text` (mesmo navy da seção anterior, continuidade). Padding vertical 60px. Borda-top 1px branco 10% opacity.

**`[Colunas — 4 colunas desktop / 2x2 mobile]`**

**ProECG:**
- Sobre
- Como Funciona
- Contato

**Legal:**
- Termos de Uso
- Política de Privacidade
- LGPD

**Suporte:**
- Central de Ajuda
- suporte@proecg.com.br
- WhatsApp

**Social:**
- Instagram
- LinkedIn

**`[Linha final — bottom bar]`**
```
© 2026 ProECG. Todos os direitos reservados.
Ferramenta de apoio à decisão clínica — não substitui avaliação médica.
```
Font-size 12px. Cor: branco 40% opacity. Centralizado.

> **Nota:** O disclaimer no footer é obrigatório e deve aparecer em toda página do sistema.

---

## RESUMO — SEÇÕES E CONVERSÃO

| # | Seção | Objetivo | CTA |
|---|-------|----------|-----|
| 0 | Navbar | Navegação + CTA sempre visível | Assinar Agora |
| 1 | Hero | Captar atenção, explicar o produto, primeira conversão | Começar Agora — R$197/mês |
| 2 | Social Proof (Números) | Credibilizar com dados | — |
| 3 | Como Funciona (3 passos) | Reduzir incerteza sobre o fluxo | — |
| 4 | Benefícios (zig-zag) | Vender por valor/resultado | — |
| 5 | Exemplo de Laudo | "Show don't tell" — ver o produto | — |
| 6 | Pricing | Converter — escolher plano | Assinar [Plano] |
| 7 | Segurança & Compliance | Remover objeções legais/regulatórias | — |
| 8 | Testimonials | Prova social qualificada | — |
| 9 | FAQ | Remover últimas dúvidas | — |
| 10 | CTA Final | Última oportunidade de conversão | Assinar o ProECG |
| 11 | Footer | Institucional + disclaimer | — |

---

## GATILHOS MENTAIS APLICADOS

| Gatilho | Onde | Como |
|---------|------|------|
| **Velocidade** | Hero, Bloco 1, Exemplo de laudo | "em segundos", cronômetro, "12s" |
| **Autoridade** | Hero, Bloco 3, Testimonials, Compliance | "cardiologistas", CRM, pipeline técnico |
| **Prova Social** | Números, Testimonials | Métricas + depoimentos com CRM |
| **Simplicidade** | Hero, Como funciona | "tirar uma foto", 3 passos |
| **Identificação** | Toda a copy | Linguagem médica: "plantão", "jaleco", "prontuário" |
| **Resultado** | Benefícios, Laudo | O médico vê o output real antes de comprar |
| **Segurança** | Compliance, FAQ | LGPD, dados temporários, disclaimer |
| **Ancoragem** | Pricing | R$267 → R$227 → R$197 (olho vai pro menor) |
| **Urgência suave** | CTA final | "O próximo ECG não precisa te deixar em dúvida" |
| **Risco zero** | Sub-CTAs, FAQ | "Cancele quando quiser", "sem multa" |
| **Escassez (implícita)** | Badge "~30 diagnósticos" | Cobrir diagnósticos críticos = necessidade real |
| **Pertencimento** | Testimonials | "Médicos como você já usam" |

---

## NOTAS PARA IMPLEMENTAÇÃO (Claude Code)

1. **Framework:** Next.js 14+ (App Router) com Tailwind CSS. Página em `src/app/(marketing)/page.tsx`.
2. **Componentes:** Criar componentes separados para cada seção (`Hero.tsx`, `HowItWorks.tsx`, `Benefits.tsx`, `Pricing.tsx`, `FAQ.tsx`, `Testimonials.tsx`, `CTAFinal.tsx`, `Footer.tsx`).
3. **Animações:** Usar Framer Motion para scroll reveals (`whileInView`) e stagger animations. CSS animations para marquee e glow.
4. **Fontes:** Importar via Google Fonts ou next/font. Definir no `tailwind.config.ts`.
5. **Responsividade:** Mobile-first. Breakpoints: `sm:640px`, `md:768px`, `lg:1024px`.
6. **Imagens/Mockups:** Criar com elementos HTML/CSS estilizados (não depender de imagens externas no MVP). Ou usar SVGs inline para o mockup do celular.
7. **Accordion do FAQ:** Usar `details/summary` nativo ou componente Radix/shadcn.
8. **Scroll suave:** `scroll-behavior: smooth` no `html`. Links da navbar fazem scroll até a seção correspondente (`#como-funciona`, `#beneficios`, `#planos`, `#faq`).
9. **SEO:** Meta tags com título "ProECG — Laudo de ECG com IA em Segundos", description, og:image com mockup do produto.
10. **Performance:** Lazy load em seções abaixo da dobra. Priorizar LCP na hero (mockup + headline).
