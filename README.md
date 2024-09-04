# Les problèmes qui vont surement resurgir à un moment ou un autre

## Spacy Dummy Layer

Étant donnée que j'ai fait quelques modification dans le module visualkeras que l'on peut trouver en open source sur git, certaines fonctionnalités de mon code peuvent ne pas fonctionnées.

### Problème d'une variable qui n'existe pas 
La modification principale que j'ai fait dans le code de visulakeras est le fait d'ajouter un padding sur un des côté pour que ma visualisation soit plus esthétique. De ce fait, j'ai dû changer
quelque partie du code source du module, et donc j'ai rajouter quelque chose dans les dimension des box par rapport au module de base

### Problème de call de la couche
Un autre problème que j'ai réussi à résoudre est le fait de devoir appeler la couche dans une architecture de couche. Au final j'utilise add car c'est plus cohérent avec la modulabilité que je 
souhaite avoir au niveau du code, cependant j'avais créer la fonction `call()` pour pouvoir directement appeler la couche dans une architecture séquentielle.

### Problème avec la clé trainable lors du chargement du modèle
En effet lorsque j'ai effectué mon changement pour pouvoir tester un modèle déjà entraîné j'ai eu un problème avec la clé trainable qui je pense se met automatiquement dans la sérialisation de la couche lorsqu'on enregistre un modèle. Hors la fonction `from_config` de la class `SpacyDummyLayer` ne prend pas la paramètre trainable dans la config, et cela fait que l'on a un problème lorsque l'on charge la couche, car le from_config n'est pas fait pour supporter cette clé `trainable` qui est ajoutée à chaque couche sérialisée, donc j'ai juste ajouté une fonction qui permet de se débarasser du paramètre trainable.


## Créer un module téléchargeable pour d'autres personnes

**Avant de lire ce qui va suivre sachez que la chose suivant est aussi possible et peut-être plus simple pour le créateur:**
Il est aussi possible de juste mettre le module modifier sur un git et de faire un git clone. Cependant c'est plus contraignant pour l'utilisateur.


Pour que d'autres personnes puissent télécharger et installer votre module Python à l'aide de pip, vous devez le publier sur le Python Package Index (PyPI). Voici les étapes générales à suivre :

    Organisez votre projet Python : Assurez-vous que votre projet est bien structuré et contient un fichier setup.py. Ce fichier contient des informations sur votre projet, telles que son nom, sa version, ses dépendances, etc.

    Créez un compte PyPI : Vous devez créer un compte sur PyPI si vous n'en avez pas déjà un.

    Préparez votre package : Avant de publier votre package, assurez-vous que toutes les dépendances sont correctement spécifiées dans le fichier setup.py. Vous devrez également inclure un fichier README décrivant votre projet, ainsi qu'un fichier LICENSE décrivant les conditions de licence de votre code.

    Enregistrez-vous avec twine : twine est un outil que vous utiliserez pour publier votre package sur PyPI. Vous pouvez l'installer en utilisant pip install twine. Ensuite, utilisez twine pour vous connecter à votre compte PyPI.

    Créez une version distribuable de votre package : Utilisez python setup.py sdist bdist_wheel pour créer une distribution de votre package.

    Publiez votre package : Utilisez twine upload dist/* pour téléverser votre package sur PyPI.

Une fois que vous avez suivi ces étapes, votre package sera disponible sur PyPI, et d'autres personnes pourront l'installer en utilisant simplement pip install <nom-de-votre-package>. Assurez-vous de bien documenter votre package et de suivre les meilleures pratiques pour que d'autres développeurs puissent l'utiliser facilement.



