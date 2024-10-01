% Template LaTeX para trabalhos acadêmicos
% 
% Pontifícia Universidade Católica de São Paulo - PUCSP
% Departamento de Computação
% Maio de 2019
% João E. L. Fouyer (jfouyer@gmail.com)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{abntpuc}
\LoadClass[a4paper, 12pt]{report}
\RequirePackage[brazil]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage[a4paper]{geometry}
\RequirePackage{lmodern}
\RequirePackage{fancyhdr}
\RequirePackage{chngcntr}
\RequirePackage{caption}
\RequirePackage{pgffor}
\RequirePackage{lastpage}
\RequirePackage{tabularx}
\RequirePackage{enumerate}
\RequirePackage{float}
\RequirePackage{titlesec}
\RequirePackage{tocloft,etoolbox}

\usepackage[alf]{abntcite}
\usepackage[pdftex]{graphicx}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[table,xcdraw]{xcolor}
\usepackage{etoolbox}
\usepackage{algpseudocode,algorithm}
\usepackage{makecell}
\usepackage{indentfirst}
\usepackage{nameref}
\usepackage{amssymb}
\usepackage{verbatim}

\patchcmd{\l@chapter}{#1}{\MakeUppercase{#1}}{}{}
\patchcmd{\l@subsection}{#1}{\normalfont{#1}}{}{}
\patchcmd{\l@subsubsection}{#1}{\normalfont{\emph{#1}}}{}{}


\geometry{top=3cm,bottom=2cm,left=3cm,right=2cm}

\setlength{\parindent}{1.25cm}
\setlength{\baselineskip}{1.5\baselineskip}
\captionsetup[figure]{labelsep=endash}
\captionsetup[table]{labelsep=endash}
\pagestyle {empty}
\pagestyle {myheadings}


%-- Configuração do cabeçalho
\setlength{\headheight}{15pt}
\font\cabecalho=cmr12 at 9pt
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\emph{\cabecalho\nouppercase\leftmark}}
\fancyhead[R]{\cabecalho\thepage}
\fancypagestyle{plain}{
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[R]{\cabecalho\thepage}
}



% --- Nome da universidade ---
\providecommand{\imprimiruniversidade}{}
\newcommand{\universidade}[1]{\renewcommand{\imprimiruniversidade}{#1}}

% --- Nome da faculdade ---
\newcommand{\imprimirfaculdade}{}
\newcommand{\faculdade}[1]{\renewcommand{\imprimirfaculdade}{#1}}

% --- Nome do departamento ---
\newcommand{\imprimirdepartamento}{}
\newcommand{\departamento}[1]{\renewcommand{\imprimirdepartamento}{#1}}

% --- Nome do curso ---
\newcommand{\imprimircurso}{}
\newcommand{\curso}[1]{\renewcommand{\imprimircurso}{#1}}

% --- Titulo do trabalho ---
\newcommand{\imprimirtitulo}{}
\newcommand{\titulo}[1]{\renewcommand{\imprimirtitulo}{#1}}

% --- Subtitulo do trabalho ---
\newcommand{\imprimirsubtitulo}{}
\newcommand{\subtitulo}[1]{\renewcommand{\imprimirsubtitulo}{#1}}

% --- Cidade ---
\newcommand{\imprimircidade}{}
\newcommand{\cidade}[1]{\renewcommand{\imprimircidade}{#1}}

% --- Ano ---
\newcommand{\imprimirano}{}
\newcommand{\ano}[1]{\renewcommand{\imprimirano}{#1}}

% --- Tipo de trabalho: Monografia, Dissertacao, Tese ---
\newcommand{\imprimirtrabalho}{}
\newcommand{\trabalho}[1]{\renewcommand{\imprimirtrabalho}{#1}}

% --- Grau do trabalho: Bacharel em..., Mestre em..., Doutor em... ---
\newcommand{\imprimirgrau}{}
\newcommand{\grau}[1]{\renewcommand{\imprimirgrau}{#1}}

% --- Nome do orientador ---
\newcommand{\imprimirorientador}{}
\newcommand{\orientador}[1]{\renewcommand{\imprimirorientador}{#1}}

% --- Nome do 1o avaliador ---
\newcommand{\imprimiravaliadorA}{}
\newcommand{\avaliadorA}[1]{\renewcommand{\imprimiravaliadorA}{#1}}

% --- Nome do 2o avaliador ---
\newcommand{\imprimiravaliadorB}{}
\newcommand{\avaliadorB}[1]{\renewcommand{\imprimiravaliadorB}{#1}}

% --- Data completa: dia, mes e ano ---
\newcommand{\imprimirdatacompleta}{}
\newcommand{\datacompleta}[1]{\renewcommand{\imprimirdatacompleta}{#1}}

% --- Preambulo Customizável ---
\newcommand{\imprimirpreambulo}{}
\newcommand{\preambulo}[1]{\renewcommand{\imprimirpreambulo}{#1}}



% --- Palavras chaves para Ficha Catalográfica ---
\providecommand{\imprimirchaves}{}
\newcommand{\palavraschave}[5]{\renewcommand{\imprimirchaves}{\hspace{0.6cm} 1. #1. \hspace{0.05cm} 2. 
#2. \hspace{0.05cm} 3. #3. \hspace{0.05cm} 4. #4}}

\font\customtitle=cmr12 at 14pt %-- Opcional para aumentar o tamanho do titulo na capa.

% --- CAPA ---
\newcommand{\capa}{
    \thispagestyle{empty}
	\begin{center}
	    \MakeUppercase{\imprimiruniversidade}\par
	    \MakeUppercase{\imprimirfaculdade}\par
	    \MakeUppercase{\imprimirdepartamento}\par
	\end{center}
	
	\vspace{6.5cm}
	
	\begin{center}
		\MakeUppercase{\customtitle\imprimirtitulo}\par
	    \imprimirsubtitulo 
	\end{center}
	
	\vspace{4cm}
	
	\begin{flushright}
	    \begin{minipage}{.3\textwidth}
            \begin{flushright}
                \renewcommand\and{\\}
                \@author
            \end{flushright}
        \end{minipage}
	\end{flushright}
	\vfill
	\begin{center}
		\imprimircidade\par 
		\imprimirano 
	\end{center}
	\newpage
}

% --- FOLHA DE ROSTO ---
\newcommand{\folharosto}{
	\thispagestyle{empty}
	\begin{center}
	    \renewcommand\and{, }
		\MakeUppercase{\@author}
	\end{center}
	\vfill
	\begin{center}
		\MakeUppercase{\customtitle\imprimirtitulo}\par
		\imprimirsubtitulo
	\end{center}
	\vfill
	\begin{flushright}
		\begin{minipage}{9cm}
		\imprimirpreambulo\par
		\vspace{1cm}
		Orientador: {\imprimirorientador}
		\end{minipage}
	\end{flushright}
	\vfill
	\begin{center}
		\imprimircidade\par
		\imprimirano
	\end{center}
	\newpage
}

% --- FOLHA DE APROVACAO ---
\newcommand{\folhaaprovacao}{
	\thispagestyle{empty}
	\begin{center}
	    \renewcommand\and{, }
		\MakeUppercase{\@author}
	\end{center}
	\vfill 
	\begin{center}
		\MakeUppercase{\customtitle\imprimirtitulo}\par
		\imprimirsubtitulo
	\end{center}
	\vfill
	\begin{flushright}
		\begin{minipage}{9cm}
			\imprimirpreambulo\par
		\end{minipage}
	\end{flushright}
	\vfill
	\begin{center}
		\rule{8cm}{0.01cm}\par 
		\imprimirorientador 
	\end{center}
	\vfill
	\begin{center}
		\rule{8cm}{0.01cm}\par 
		\imprimiravaliadorA
	\end{center}
	\vfill 
	\begin{center}
		\rule{8cm}{0.01cm}\par
		\imprimiravaliadorB 
	\end{center}
	\vfill
	\begin{center}
		\imprimircidade, \imprimirdatacompleta
	\end{center}
	\newpage 
}

% --- DEDICATORIA ---
\newcommand{\dedicatoria}[1]
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip} 
	\begin{flushright}
		\begin{minipage}{12cm}
			{#1} 
		\end{minipage}
	\end{flushright}
\end{newpage}
}

% --- EPIGRAFE ---
\newcommand{\imprimirepigrafe}[1]
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip} 
	\begin{flushright}
		\begin{minipage}{12cm}
			\imprimirepigrafequote
		\end{minipage}
	\end{flushright}
\end{newpage}
}

% --- AGRADECIMENTOS ---
\newcommand{\agradecimentos}[1]
{
\begin{newpage}
	\thispagestyle{empty}	
	\setlength{\parindent}{1.25cm}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{AGRADECIMENTOS} \\ [1.5\baselineskip]
	\end{center}
	{#1}
\end{newpage}
}

% --- RESUMO ---
\newcommand{\resumo}[2]
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{RESUMO} \\ [1.5\baselineskip]
	\end{center}
	\noindent{#1}
	\vspace{1.5\baselineskip}
	\noindent Palavras-chave: {#2}
\end{newpage}
}

% --- ABSTRACT ---
\renewcommand{\abstract}[2]
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{ABSTRACT} \\ [1.5\baselineskip]
	\end{center}
	\noindent{\setlength{\baselineskip}{1.5\baselineskip}#1}
	\vspace{1.5\baselineskip}	
	\noindent Keywords: {#2}
\end{newpage}
}

% \patchcmd{\l@chapter}{#1}{\MakeUppercase{#1}}{}{}
% \patchcmd{\l@subsection}{#1}{\normalfont{#1}}{}{}
% \patchcmd{\l@subsubsection}{#1}{\normalfont{\emph{#1}}}{}{}

% --- SUMARIO ---
\newcommand{\sumario}
{
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.0\baselineskip}
	\begin{center}
		\textbf{SUMÁRIO}
	\end{center}
	\@starttoc{toc}
	\newpage
    \setlength{\baselineskip}{1.5\baselineskip}
}


\usepackage{tocloft}
% --- LISTA DE FIGURAS ---
\usepackage{chngcntr} 
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}
\renewcommand{\cftfigpresnum}{FIGURA \ }
\setlength{\cftfignumwidth}{6em}
\renewcommand{\cftfigaftersnum}{ \ -- \ }
\renewcommand{\thefigure}{\arabic{figure}}

\newcommand{\listafiguras}
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{LISTA DE FIGURAS} \\ [1.5\baselineskip]
	\end{center}
	\@starttoc{lof}
\end{newpage}
}

% --- LISTA DE TABELAS ---
\renewcommand{\cfttabpresnum}{TABELA \ }
\setlength{\cfttabnumwidth}{6em}
\renewcommand{\cfttabaftersnum}{ \ -- \ }
\renewcommand{\thetable}{\arabic{table}}
\newcommand{\listatabelas}
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{LISTA DE TABELAS} \\ [1.5\baselineskip]
	\end{center}
	\@starttoc{lot}
\end{newpage}
}


\newcommand{\acronymlistname}{LISTA DE SIGLAS}
\newlistof{acronym}{loac}{\acronymlistname}
\newcommand{\acronym}[2]{\refstepcounter{acronym}\addtocontents{loac}{\protect #1 \ -- \ #2 \\}}
\newcommand{\sigla}[2]{\acronym{#1}{#2}}
\newcommand{\listasiglas}
{
\begin{newpage}
	\thispagestyle{empty}
	\setlength{\baselineskip}{1.5\baselineskip}
	\begin{center}
		\textbf{\acronymlistname} \\ [1.5\baselineskip]
	\end{center}
	\@starttoc{loac}
        \newpage
\end{newpage}
}
%% FIM LISTA DE EXEMPLOS
\setcounter{tocdepth}{3}
\setlength{\cftbeforechapskip}{\baselineskip}
%\renewcommand{\cftchapfont}{\upshape\bfseries}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftsecindent}{0em}
\renewcommand{\cftsecfont}{\bfseries}
\setlength{\cftsubsecindent}{0em}
\renewcommand{\cftsubsecfont}{\itshape}
\setlength{\cftsubsubsecindent}{0em}
\renewcommand{\cftsubsubsecfont}{\bfseries}

% --- Titulo de capitulo ---
\newcommand{\capitulo}[1]{\chapter{#1} \thispagestyle{myheadings}}
\renewcommand{\@makechapterhead}[1]{\noindent{\MakeUppercase{\textbf{\thechapter~#1}}\par}\vspace{0.5cm}}

% --- Titulo de secao ---
\newcommand{\secao}[1]{\section{#1}}
\renewcommand{\section}{\@startsection{section}
{0}
{0mm}
{0.5\baselineskip}
{0.5\baselineskip}
{\noindent\bfseries}}

% --- Titulo de subsecao ---
\newcommand{\subsecao}[1]{\subsection{#1}}
\renewcommand{\subsection}{\@startsection
{subsection}
{1}
{0mm}
{0.5\baselineskip}
{0.5\baselineskip}
{\noindent\bfseries}}

% --- Titulo de subsubsecao ---
\newcommand{\subsubsecao}[1]{\subsubsection{#1}}
\renewcommand{\subsubsection}{\@startsection
{subsubsection}
{2}
{0mm}
{0.5\baselineskip}
{0.5\baselineskip} 
{\noindent\itshape}} 

\usepackage[alf]{abntcite}
% --- REFERENCIAS ---
\def\@makeschapterhead#1{
        \thispagestyle{empty}
	\begin{center} \textbf{#1}\par \end{center}
        % \addcontentsline{toc}{chapter}{\bibname}
	\vspace{1.5\baselineskip}
}

\newcommand{\referencias}[1]
{
	\thispagestyle{myheadings}
        \setlength{\baselineskip}{0.73\baselineskip}
	\renewcommand{\bibname}{REFERÊNCIAS}
	\bibliographystyle{abnt-alf}
	\bibliography{#1}
}