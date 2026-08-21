import re

with open('docs/protodb.tex', 'r') as f:
    content = f.read()

# 1. Update Section 4.5: Flat Index to mention Deletion
deletion_text = r"""
\subsection{Deletion}

Vectors can be marked as deleted using a tombstone mechanism. 
The \texttt{VecPageHeader} includes a 128-byte \texttt{deleted\_bitmap} which tracks up to 1024 vectors per page.
When \texttt{flat\_index\_delete} is called, the index locates the vector's page and slot, and sets the corresponding bit in the bitmap. 
The search path is updated to check this bitmap via \texttt{vec_page_is_deleted(page, i)} and skips vectors that have been marked as deleted. This avoids costly in-place compaction while safely removing vectors from search results.
"""
content = content.replace(r"\subsection{Complexity}", deletion_text + "\n" + r"\subsection{Complexity}")

# 2. Add Chapter 5: IVF Index and Chapter 6: Persistence before \end{document}
new_chapters = r"""
% ═════════════════════════════════════════════════════════════════════════════
%  CHAPTER 5: INVERTED FILE (IVF) INDEX
% ═════════════════════════════════════════════════════════════════════════════
\chapter{Inverted File (IVF) Index}

\section{Overview}
While the Flat Index provides exact nearest neighbor search, its $O(n)$ search complexity becomes a bottleneck for larger datasets. To address this, ProtoDB implements an Inverted File (IVF) Index, a classic approximate nearest neighbor (ANN) algorithm.

\section{Architecture}
The IVF index partitions the vector space into \texttt{nlist} Voronoi cells, each defined by a centroid. 
Vectors are assigned to the cell of their nearest centroid. During search, the query is compared against all centroids, and only the vectors in the \texttt{nprobe} closest cells are scanned.

\subsection{Centroid Training}
Before inserting vectors, the index must be trained. A subset of vectors is used to compute the centroids using K-Means clustering. 

\subsection{Partitions}
Each centroid has an associated partition, which is essentially a dynamically sized array of vector page IDs. Vectors are inserted into the pages of their respective partition. 

\section{Performance}
By tuning \texttt{nlist} and \texttt{nprobe}, the user trades off recall accuracy for search speed. In our benchmarks, an IVF index with \texttt{nlist=32} and \texttt{nprobe=4} scans only ~12\% of the dataset, increasing search throughput from ~1,300 QPS (Flat) to over ~19,500 QPS on 128-dimensional vectors.

% ═════════════════════════════════════════════════════════════════════════════
%  CHAPTER 6: PERSISTENCE AND STREAM PAGES
% ═════════════════════════════════════════════════════════════════════════════
\chapter{Persistence and Stream Pages}

\section{The Need for Persistence}
Indexes in ProtoDB originally built their metadata (e.g., page ID arrays for Flat, centroids and partition arrays for IVF) strictly in memory. While the underlying vector pages were written to disk via the buffer pool, the structural links to find them would be lost on restart.

\section{Stream Pages}
To serialize dynamically sized arrays (like \texttt{idx->page\_ids}), we introduced \texttt{PAGE\_TYPE\_STREAM}.
A Stream Page allows storing arbitrary byte streams across multiple linked pages on disk.
The \texttt{StreamPageHeader} contains a \texttt{next\_page\_id} and \texttt{data\_size}.
This acts as a linked list of pages, enabling us to serialize index metadata that exceeds the 8KB limit of a single page.

\section{Index Serialization}
Each index type implements a \texttt{save} and \texttt{load} routine:
\begin{itemize}
    \item \textbf{Flat Index}: Serializes a \texttt{FlatIndexMeta} header followed by the array of vector page IDs.
    \item \textbf{IVF Index}: Iterates through all partitions, saves them using stream pages, and then serializes the \texttt{IvfIndexMeta}, centroids, and the array of partition root page IDs.
\end{itemize}

The root page ID of the main index stream is stored in the \texttt{MetaPageHeader} (page 0), allowing ProtoDB to seamlessly rebuild the entire database state upon restart.

"""

content = content.replace(r"\end{document}", new_chapters + "\n" + r"\end{document}")

with open('docs/protodb.tex', 'w') as f:
    f.write(content)
