# recommender-system
Projet  d'étude système de recommendation en utilisant filtrage collaboratif
***
## Methode et approche utilisé

Pour un système de recommandation basé sur SVD, voici une stratégie très simple : trouver l'utilisateur le plus similaire à l'aide des matrices bidimensionnelles ci-dessus avec l'un des algorithmes de calcul de similarité et comparer ses éléments à celui du nouvel utilisateur ; prenez les articles que l'utilisateur similaire a évalués et que le nouvel utilisateur n'a pas évalués et retournez-les pour le nouvel utilisateur. De la même manière, pour un nouvel élément, recherchez l'élément le plus similaire à l'aide des matrices bidimensionnelles ci-dessus avec l'un des algorithmes de calcul de similarité et comparez l'élément similaire évalué par les utilisateurs avec le nouvel élément ; prendre les utilisateurs qui notent un article similaire mais pas le nouvel article et renvoyer les notes pour le nouvel article
- User-based similarité
  Dans la recommandation SVD, afin de décider si deux utilisateurs sont similaires,
  la matrice réduite Uki est utilisée[1]. Dans la (mxk)Ukmatrice, chaque ligne représente un utilisateur.
  Plus deux lignes sont similaires, plus les utilisateurs se ressemblent. Une étape critique de l'algorithme SVD consiste à calculer la similarité entre les utilisateurs, puis à sélectionner les utilisateurs les plus similaires. Il existe plusieurs façons de calculer la similarité entre les utilisateurs. Voici deux de ces méthodes : la similarité basée sur le cosinus et la comparaison des distances euclidiennes dans l'espace à k dimensions.

  - cosinus simularité:
  La similarité de deux vecteurs lignes pourrait être calculée avec la similarité en cosinus afin de décider si deux utilisateurs sont similaires ou non. Formellement, la similarité entre les utilisateurs i et ji est indiquée en (2).

  Pour trouver l'utilisateur le plus similaire à l'utilisateur 1, les distances entre le point numéro 1 et tous les points doivent être calculées et l'utilisateur qui a la plus petite distance au point 1 est l'utilisateur le plus similaire à l'utilisateur 1.

  METHODE DE REQUETAGE POUR(Utilisateur=x, Item= y, Note = ?)

  Demande de recommandation (Utilisateur= x, Item= y, Note =?)
    1.Trouvez les utilisateurs qui ont noté Item = y à partir de la matrice d'origine A.
    2.Trouvez l'utilisateur le plus similaire à Utilisateur = x parmi les utilisateurs qui ont noté Item = y en utilisant la matrice réduite Uk.
    3.Obtenez la note de l'utilisateur le plus similaire à Item = y à partir
   de la matrice d'origine A et attribuez-la à User = x , Item= y

  Pour la deuxième partie de l'algorithme, si l'utilisateur = x est un utilisateur déjà existant,
  il existe dans la matrice réduite Ukas une ligne. Si l'utilisateur = x est un nouvel utilisateur,
  avant de commencer les contrôles de similarité, l'utilisateur doit être projeté de n dimensions à k dimensions.
  Soit la notation du nouveau vecteur utilisateur Nu(1xn). La projection P vers la matrice réduite Ukis
  est faite par la formule[1]

- Comment recommender le produits pour un utilisateur x?
  - Recherchez le produit que l'utilisateur similaire y a notes et supprimer le produit que x a deja noté


  def SGD(self,n_factor=None, n_epoches=None, alpha=0.025):

      data, dict_users, dict_items = df.encode_ids(self.user_item)
      self.n_factor = self.low_rank_approximate() # number factor with low-rank approximation
      n_users = len(self.all_user_id) #numbers of unique users
      n_items = len(self.all_item__id) #numbers of unique items

      # Random initialize the user and item factor
      p = np.random.normal(0, .1, (n_users, self.n_factor))
      q = np.random.normal(0, .1, (n_items, self.n_factor))
      # p = self.left_singular_vector
      # q = self.right_singular_vector.transpose()
      # print(p[1])
      # print(q[1])
      #Minimisation and optimization procedure
      for epoches in range(n_epoches):
          print(f"tour numero: {epoches}")
          # break
          #loop over the rows in data
          for index in range(data.shape[0]):
              row = data.iloc[[index]]
              for column in list(row.columns):
                  u = int(row.index.values)                #current user_id = position in the p vector
                  i = int(column)                          #current item_id = position in the q vector
                  r_ui = float(data.loc[u, i])        #rating associate to the couple (user u, item i)
                  # print(f"u={u}, i={i}, rating={r_ui}")
                  # Error between the predicted rating (p_u * q_i) and the know rating r_ui
                  error = r_ui - np.dot(np.dot(p[u], self.fill_sigma(nbr_factor=self.n_factor)), q[i].transpose())

                  # Update vector p_u and q_i using partial deverive
                  p_old = p[u]
                  p[u] = (p[u] + alpha * error * q[i]).astype('float')
                  q[i] = (q[i] + alpha * error * p_old).astype('float')
                  # print(p[u])
                  # break

      return p, q
